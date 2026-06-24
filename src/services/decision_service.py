# -*- coding: utf-8 -*-
"""Build and render daily actionable decision reports from analysis results."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


CATEGORY_EXECUTABLE = "today_executable"
CATEGORY_WAIT = "wait_for_confirmation"
CATEGORY_AVOID = "avoid_or_remove"

CATEGORY_LABELS = {
    CATEGORY_EXECUTABLE: "今日可执行",
    CATEGORY_WAIT: "等待确认",
    CATEGORY_AVOID: "禁止追高/剔除",
}


def write_decision_report(
    results: list[Any],
    *,
    output_path: str | Path = "reports/daily_decision_report.md",
    report_type: str = "daily_action_list",
    template_path: str | Path = "templates/decision_report_markdown.j2",
) -> dict[str, Any]:
    report = build_decision_report(results, report_type=report_type)
    markdown = render_decision_report(report, template_path=template_path)

    markdown_path = Path(output_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    json_path = markdown_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    report["files"] = {
        "markdown": str(markdown_path),
        "json": str(json_path),
    }
    return report


def build_decision_report(results: list[Any], *, report_type: str = "daily_action_list") -> dict[str, Any]:
    items = [_build_decision_item(result) for result in results or [] if result]
    groups = {
        CATEGORY_EXECUTABLE: sorted(
            [item for item in items if item["category"] == CATEGORY_EXECUTABLE],
            key=lambda x: x["score"],
            reverse=True,
        ),
        CATEGORY_WAIT: sorted(
            [item for item in items if item["category"] == CATEGORY_WAIT],
            key=lambda x: x["score"],
            reverse=True,
        ),
        CATEGORY_AVOID: sorted(
            [item for item in items if item["category"] == CATEGORY_AVOID],
            key=lambda x: x["score"],
            reverse=True,
        ),
    }
    return {
        "report_type": report_type,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(items),
        "groups": groups,
        "summary": {key: len(value) for key, value in groups.items()},
    }


def render_decision_report(report: dict[str, Any], *, template_path: str | Path) -> str:
    path = Path(template_path)
    if path.exists():
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape

            env = Environment(
                loader=FileSystemLoader(str(path.parent)),
                autoescape=select_autoescape(default=False),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            template = env.get_template(path.name)
            return template.render(report=report, category_labels=CATEGORY_LABELS)
        except Exception:
            pass
    return _render_markdown_fallback(report)


def _build_decision_item(result: Any) -> dict[str, Any]:
    dashboard = _get_value(result, "dashboard", {}) or {}
    core = dashboard.get("core_conclusion") if isinstance(dashboard, dict) else {}
    battle_plan = dashboard.get("battle_plan") if isinstance(dashboard, dict) else {}
    sniper_points = battle_plan.get("sniper_points") if isinstance(battle_plan, dict) else {}
    position_strategy = battle_plan.get("position_strategy") if isinstance(battle_plan, dict) else {}
    checklist = battle_plan.get("action_checklist") if isinstance(battle_plan, dict) else []
    risk_flags = _collect_risk_flags(result)
    decision_type = str(_get_value(result, "decision_type", "hold") or "hold").lower()
    operation_advice = str(_get_value(result, "operation_advice", "观望") or "观望")
    score = _to_int(_get_value(result, "sentiment_score", 0)) or 0
    category = _resolve_category(decision_type, operation_advice, score, risk_flags)
    position_size = _resolve_position_size(decision_type, score, risk_flags)

    return {
        "symbol": _get_value(result, "code", "UNKNOWN"),
        "name": _get_value(result, "name", ""),
        "decision": _resolve_decision(decision_type, operation_advice, score, risk_flags),
        "category": category,
        "operation_advice": operation_advice,
        "action_instruction": _resolve_action_instruction(category, operation_advice, core),
        "confidence_level": _get_value(result, "confidence_level", "中"),
        "score": score,
        "entry_condition": _resolve_entry_condition(core, checklist, operation_advice),
        "entry_price": _clean_point(sniper_points.get("ideal_buy") or sniper_points.get("entry_price")),
        "secondary_entry": _clean_point(sniper_points.get("secondary_buy")),
        "stop_loss": _clean_point(sniper_points.get("stop_loss")),
        "target_price": _clean_point(
            sniper_points.get("target_price") or sniper_points.get("target") or sniper_points.get("take_profit")
        ),
        "position_size": position_size,
        "position_text": _resolve_position_text(position_strategy, position_size),
        "invalid_condition": _resolve_invalid_condition(sniper_points, risk_flags),
        "risk_flags": risk_flags,
        "quant_confidence": _resolve_quant_confidence(score, risk_flags),
        "core_conclusion": _resolve_core_conclusion(result, core),
    }


def _resolve_category(decision_type: str, operation_advice: str, score: int, risk_flags: list[str]) -> str:
    advice = operation_advice.strip()
    if decision_type == "sell" or any(word in advice for word in ("卖出", "减仓", "剔除")):
        return CATEGORY_AVOID
    if any(word in advice for word in ("严禁追高", "禁止追高")):
        return CATEGORY_AVOID
    if decision_type == "buy" and not _has_blocking_risk(risk_flags):
        try:
            from src.config import get_config
            from src.analyzer import _get_strategy_thresholds
            style = getattr(get_config(), "strategy_style", "conservative")
            t = _get_strategy_thresholds(style)
            threshold = t["executable_score"]
        except Exception:
            threshold = 65
        if score >= threshold:
            return CATEGORY_EXECUTABLE
    return CATEGORY_WAIT


def _resolve_decision(decision_type: str, operation_advice: str, score: int, risk_flags: list[str]) -> str:
    category = _resolve_category(decision_type, operation_advice, score, risk_flags)
    if category == CATEGORY_EXECUTABLE:
        return "execute"
    if category == CATEGORY_AVOID:
        return "avoid"
    return "wait"


def _resolve_action_instruction(category: str, operation_advice: str, core: Any) -> str:
    if isinstance(core, dict):
        position_advice = core.get("position_advice")
        if isinstance(position_advice, dict):
            no_position = str(position_advice.get("no_position") or "").strip()
            has_position = str(position_advice.get("has_position") or "").strip()
            if no_position or has_position:
                return f"空仓：{no_position or '等待'}；持仓：{has_position or '按计划持有'}"
    if category == CATEGORY_EXECUTABLE:
        return f"按条件执行：{operation_advice}"
    if category == CATEGORY_AVOID:
        return f"不新开仓，按风控处理：{operation_advice}"
    return f"等待确认，不主动追单：{operation_advice}"


def _resolve_position_text(position_strategy: Any, position_size: float | None) -> str:
    if isinstance(position_strategy, dict):
        suggested = str(position_strategy.get("suggested_position") or "").strip()
        if suggested:
            return suggested
    if position_size is None:
        return "观望/不新增仓位"
    if position_size <= 0:
        return "0成"
    return f"{position_size:.0%}"


def _resolve_entry_condition(core: Any, checklist: Any, operation_advice: str) -> str:
    if isinstance(core, dict):
        no_position = core.get("position_advice", {}).get("no_position") if isinstance(core.get("position_advice"), dict) else None
        if no_position:
            return str(no_position)
    if isinstance(checklist, list):
        for item in checklist:
            text = str(item).strip()
            if text and not text.startswith("❌"):
                return text
    return operation_advice or "等待满足买入条件"


def _resolve_invalid_condition(sniper_points: dict[str, Any], risk_flags: list[str]) -> str:
    stop_loss = _clean_point(sniper_points.get("stop_loss"))
    if stop_loss:
        return f"跌破止损条件：{stop_loss}"
    if risk_flags:
        return str(risk_flags[0])
    return "关键条件失效或量能衰竭"


def _resolve_position_size(decision_type: str, score: int, risk_flags: list[str]) -> float | None:
    if decision_type == "sell":
        return 0.0
    if decision_type != "buy":
        return None
    if _has_blocking_risk(risk_flags):
        return 0.1
    if score >= 80:
        return 0.2
    return 0.1


def _resolve_quant_confidence(score: int, risk_flags: list[str]) -> str:
    if _has_blocking_risk(risk_flags) or score < 45:
        return "low"
    if score >= 70:
        return "high"
    return "medium"


def _collect_risk_flags(result: Any) -> list[str]:
    flags = []
    risk_warning = _get_value(result, "risk_warning", "")
    if risk_warning:
        flags.append(str(risk_warning))
    dashboard = _get_value(result, "dashboard", {}) or {}
    intelligence = dashboard.get("intelligence") if isinstance(dashboard, dict) else {}
    alerts = intelligence.get("risk_alerts") if isinstance(intelligence, dict) else []
    if isinstance(alerts, list):
        flags.extend(str(item) for item in alerts if str(item).strip())
    return list(dict.fromkeys(flags))


def _has_blocking_risk(risk_flags: list[str]) -> bool:
    risk_text = " ".join(risk_flags).lower()
    keywords = ("严禁追高", "禁止追高", "跑输", "不一致", "高回撤", "underperformed", "inconsistent")
    return any(keyword.lower() in risk_text for keyword in keywords)


def _resolve_core_conclusion(result: Any, core: Any) -> str:
    if isinstance(core, dict) and core.get("one_sentence"):
        return str(core.get("one_sentence"))
    return str(_get_value(result, "analysis_summary", ""))


def _clean_point(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    return re.sub(r"^(理想买入点|次优买入点|止损位|目标位)[:：]\s*", "", text)


def _get_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _render_markdown_fallback(report: dict[str, Any]) -> str:
    lines = [
        f"# 每日行动清单 - {report.get('generated_at', '')}",
        "",
        f"总计：{report.get('total', 0)} 只标的",
        "",
    ]
    groups = report.get("groups") or {}
    for category, label in CATEGORY_LABELS.items():
        items = groups.get(category) or []
        lines.extend([f"## {label}（{len(items)}）", ""])
        if not items:
            lines.extend(["- 无", ""])
            continue
        for item in items:
            name = item.get("name") or item.get("symbol")
            symbol = item.get("symbol", "")
            advice = item.get("operation_advice", "—")

            if category == CATEGORY_EXECUTABLE:
                entry = item.get("entry_price") or "—"
                stop = item.get("stop_loss") or "—"
                target = item.get("target_price") or "—"
                pos = item.get("position_text") or "—"
                score = item.get("score", 0)
                lines.append(f"- 🎯 **{name}**（{symbol}）— {advice}（评分 {score}）")
                lines.append(f"  入场 {entry} ｜ 止损 {stop} ｜ 目标 {target} ｜ 仓位 {pos}")
            elif category == CATEGORY_WAIT:
                condition = item.get("entry_condition") or "—"
                score = item.get("score", 0)
                lines.append(f"- ⏳ **{name}**（{symbol}）— {advice}（评分 {score}）")
                lines.append(f"  条件：{condition}")
            else:
                reason = (item.get("risk_flags") or [None])[0] or item.get("core_conclusion") or "—"
                score = item.get("score", 0)
                lines.append(f"- 🛑 **{name}**（{symbol}）— {advice}（评分 {score}）")
                lines.append(f"  原因：{reason}")
        lines.append("")
    return "\n".join(lines)
