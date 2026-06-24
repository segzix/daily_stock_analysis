# -*- coding: utf-8 -*-
"""Helpers for loading quant backtest context into prompts and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config import get_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_json(path_value: str) -> Optional[Dict[str, Any]]:
    path = _resolve_path(path_value)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"items": data}


def _symbol_key(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text.split(".")[0]


def _matches_code(item: Dict[str, Any], code: str) -> bool:
    target = _symbol_key(code)
    for key in ("code", "symbol", "stock_code", "ticker"):
        if _symbol_key(item.get(key)) == target:
            return True
    return False


def _first_present(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("items", "results", "summaries", "backtests", "stocks"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    for key in ("by_symbol", "by_code", "symbols"):
        value = payload.get(key)
        if isinstance(value, dict):
            return [
                {**item, "symbol": item.get("symbol") or symbol}
                for symbol, item in value.items()
                if isinstance(item, dict)
            ]

    items = []
    meta_keys = {
        "strategy",
        "start_date",
        "end_date",
        "total",
        "success",
        "failed",
        "source_file",
        "generated_at",
        "created_at",
    }
    for key, value in payload.items():
        if key in meta_keys or not isinstance(value, dict):
            continue
        if any(metric_key in value for metric_key in ("metrics", "total_return_pct", "sharpe_ratio", "status")):
            items.append({**value, "symbol": value.get("symbol") or key})
    return items


def _normalize_quant_item(item: Dict[str, Any]) -> Dict[str, Any]:
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    best_params = item.get("best_params") if isinstance(item.get("best_params"), dict) else {}
    assessment = item.get("assessment") if isinstance(item.get("assessment"), dict) else {}

    normalized_metrics = {
        "total_return_pct": _as_float(
            _first_present(metrics, ("total_return_pct", "total_return", "return_pct", "return"))
            if metrics
            else _first_present(item, ("total_return_pct", "total_return", "return_pct", "return"))
        ),
        "benchmark_return_pct": _as_float(
            _first_present(metrics, ("benchmark_return_pct", "benchmark_return", "benchmark_pct"))
            if metrics
            else _first_present(item, ("benchmark_return_pct", "benchmark_return", "benchmark_pct"))
        ),
        "max_drawdown_pct": _as_float(
            _first_present(metrics, ("max_drawdown_pct", "max_drawdown", "drawdown_pct"))
            if metrics
            else _first_present(item, ("max_drawdown_pct", "max_drawdown", "drawdown_pct"))
        ),
        "sharpe_ratio": _as_float(
            _first_present(metrics, ("sharpe_ratio", "sharpe")) if metrics else _first_present(item, ("sharpe_ratio", "sharpe"))
        ),
        "win_rate_pct": _as_float(
            _first_present(metrics, ("win_rate_pct", "win_rate")) if metrics else _first_present(item, ("win_rate_pct", "win_rate"))
        ),
    }
    normalized_params = {
        "fast_window": _first_present(best_params, ("fast_window", "fast")) or item.get("fast_window"),
        "slow_window": _first_present(best_params, ("slow_window", "slow")) or item.get("slow_window"),
    }
    success = bool(_first_present(item, ("success", "ok", "completed"))) or str(item.get("status", "")).lower() in {
        "success",
        "completed",
        "ok",
    }
    risk_level = str(_first_present(assessment, ("risk_level", "risk")) or item.get("risk_level") or "N/A")
    total_return = normalized_metrics["total_return_pct"]
    benchmark_return = normalized_metrics["benchmark_return_pct"]
    sharpe_ratio = normalized_metrics["sharpe_ratio"]
    explicit_effective = _first_present(assessment, ("is_effective", "effective"))
    if explicit_effective is None:
        effective = success and (
            (total_return is not None and benchmark_return is not None and total_return >= benchmark_return)
            or (sharpe_ratio is not None and sharpe_ratio > 0)
        )
    else:
        effective = bool(explicit_effective)

    symbol = item.get("symbol") or item.get("code") or item.get("stock_code") or item.get("ticker") or "N/A"
    return {
        "symbol": str(symbol),
        "code": item.get("code") or str(symbol).split(".")[0],
        "name": item.get("name") or item.get("stock_name") or "",
        "status": item.get("status") or ("completed" if success else "failed"),
        "success": success,
        "metrics": normalized_metrics,
        "best_params": normalized_params,
        "assessment": {
            "risk_level": risk_level,
            "is_effective": effective,
            "conclusion": assessment.get("conclusion") or item.get("conclusion") or "",
        },
        "raw": item,
    }


def get_quant_summary_by_code(code: str, summary_path: str) -> Optional[Dict[str, Any]]:
    payload = _load_json(summary_path)
    if not payload:
        return None
    for item in _extract_items(payload):
        if _matches_code(item, code):
            return _normalize_quant_item(item)
    return None


def get_cloud_summary_by_code(code: str, summary_path: str) -> Optional[Dict[str, Any]]:
    payload = _load_json(summary_path)
    if not payload:
        return None
    for item in _extract_items(payload):
        if _matches_code(item, code):
            return item
    return None


def get_backtest_comparison_by_code(code: str, comparison_path: str) -> Optional[Dict[str, Any]]:
    payload = _load_json(comparison_path)
    if not payload:
        return None
    for item in _extract_items(payload):
        if _matches_code(item, code):
            return item
    return None


def format_quant_summary_for_prompt(summary: Optional[Dict[str, Any]]) -> str:
    if not summary:
        return ""
    metrics = summary.get("metrics") or {}
    assessment = summary.get("assessment") or {}
    return f"""

## 📈 量化回测参考

- 标的：{summary.get('symbol', 'N/A')}
- 总收益：{metrics.get('total_return_pct', 'N/A')}%
- 基准收益：{metrics.get('benchmark_return_pct', 'N/A')}%
- 最大回撤：{metrics.get('max_drawdown_pct', 'N/A')}%
- 夏普：{metrics.get('sharpe_ratio', 'N/A')}
- 胜率：{metrics.get('win_rate_pct', 'N/A')}%
- 风险等级：{assessment.get('risk_level', 'N/A')}
- 是否生效：{'是' if assessment.get('is_effective') else '否'}
"""


def format_backtest_comparison_for_prompt(
    comparison: Optional[Dict[str, Any]],
    cloud_summary: Optional[Dict[str, Any]],
) -> str:
    if not comparison and not cloud_summary:
        return ""
    return f"""

## 🔁 本地/云端回测一致性

- 对比结果：{comparison or 'N/A'}
- 云端摘要：{cloud_summary or 'N/A'}
"""


def load_quant_backtest_context(results: List[Any]) -> Dict[str, Any]:
    config = get_config()
    enabled = bool(getattr(config, "quant_backtest_enabled", False))
    summary_path = getattr(config, "quant_backtest_summary_path", "reports/stock_pool_backtest_summary.json")
    source_path = _resolve_path(summary_path)
    if not enabled:
        return {
            "quant_backtests": {
                "enabled": False,
                "status": "disabled",
                "conclusion": "量化回测未启用",
                "items": [],
                "source_file": str(source_path),
            }
        }

    payload = _load_json(summary_path)
    if not payload:
        return {
            "quant_backtests": {
                "enabled": True,
                "status": "missing",
                "conclusion": "量化回测已启用，但未找到摘要文件",
                "items": [],
                "source_file": str(source_path),
            }
        }

    wanted_codes = {_symbol_key(getattr(result, "code", "")) for result in results}
    items = [
        _normalize_quant_item(item)
        for item in _extract_items(payload)
        if not wanted_codes or any(_matches_code(item, code) for code in wanted_codes)
    ]
    items = [item for item in items if item.get("symbol")]
    success_count = sum(1 for item in items if item.get("success"))
    effective_count = sum(1 for item in items if (item.get("assessment") or {}).get("is_effective"))
    failed_count = max(len(items) - success_count, 0)
    top_by_sharpe = sorted(
        items,
        key=lambda item: (item.get("metrics") or {}).get("sharpe_ratio")
        if (item.get("metrics") or {}).get("sharpe_ratio") is not None
        else -999999,
        reverse=True,
    )
    high_risk_items = [
        item
        for item in items
        if str((item.get("assessment") or {}).get("risk_level", "")).lower() in {"high", "高", "high_risk"}
    ]
    if not items:
        status = "no_match"
        conclusion = "量化回测已启用，但摘要中没有命中本次报告股票"
    elif effective_count > 0:
        status = "effective"
        conclusion = f"量化回测生效：{effective_count}/{len(items)} 只股票通过策略验证"
    else:
        status = "ineffective"
        conclusion = "量化回测未形成有效验证结论"

    return {
        "quant_backtests": {
            "enabled": True,
            "status": status,
            "effective": effective_count > 0,
            "conclusion": conclusion,
            "strategy": payload.get("strategy") or payload.get("strategy_name") or "MA crossover",
            "start_date": payload.get("start_date") or payload.get("from") or "N/A",
            "end_date": payload.get("end_date") or payload.get("to") or "N/A",
            "total": len(items),
            "success": success_count,
            "failed": failed_count,
            "items": items,
            "top_by_sharpe": top_by_sharpe,
            "high_risk_items": high_risk_items,
            "source_file": str(source_path),
        }
    }
