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


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "ok", "success", "completed"}:
        return True
    if text in {"false", "0", "no", "n", "failed", "error", "insufficient_data"}:
        return False
    return None


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
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
        "data_source",
        "benchmark",
        "assumptions",
        "strategy_rule",
    }
    for key, value in payload.items():
        if key in meta_keys or not isinstance(value, dict):
            continue
        if any(metric_key in value for metric_key in ("metrics", "total_return_pct", "sharpe_ratio", "status")):
            items.append({**value, "symbol": value.get("symbol") or key})
    return items


def _normalize_quant_item(item: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    best_params = item.get("best_params") if isinstance(item.get("best_params"), dict) else {}
    assessment = item.get("assessment") if isinstance(item.get("assessment"), dict) else {}
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    context = item.get("backtest_context") if isinstance(item.get("backtest_context"), dict) else {}
    assumptions = context or payload.get("assumptions") if isinstance(payload.get("assumptions"), dict) else {}
    benchmark = (
        context.get("benchmark")
        if isinstance(context.get("benchmark"), dict)
        else payload.get("benchmark") if isinstance(payload.get("benchmark"), dict) else {}
    )

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
        "trade_count": _as_float(
            _first_present(metrics, ("trade_count", "trades")) if metrics else _first_present(item, ("trade_count", "trades"))
        ),
        "annualized_return_pct": _as_float(
            _first_present(metrics, ("annualized_return_pct", "annual_return_pct", "annualized_return"))
            if metrics
            else _first_present(item, ("annualized_return_pct", "annual_return_pct", "annualized_return"))
        ),
        "volatility_pct": _as_float(
            _first_present(metrics, ("volatility_pct", "annualized_volatility_pct", "volatility"))
            if metrics
            else _first_present(item, ("volatility_pct", "annualized_volatility_pct", "volatility"))
        ),
        "sample_count": _as_float(
            _first_present(metrics, ("sample_count", "sample_days", "trading_days"))
            if metrics
            else _first_present(item, ("sample_count", "sample_days", "trading_days"))
        ),
        "calmar_ratio": _as_float(
            _first_present(metrics, ("calmar_ratio", "calmar")) if metrics else _first_present(item, ("calmar_ratio", "calmar"))
        ),
        "profit_factor": _as_float(
            _first_present(metrics, ("profit_factor",)) if metrics else _first_present(item, ("profit_factor",))
        ),
        "avg_holding_days": _as_float(
            _first_present(metrics, ("avg_holding_days", "avg_hold_days", "avg_holding"))
            if metrics
            else _first_present(item, ("avg_holding_days", "avg_hold_days", "avg_holding"))
        ),
        "max_consecutive_wins": _as_float(
            _first_present(metrics, ("max_consecutive_wins",)) if metrics else _first_present(item, ("max_consecutive_wins",))
        ),
        "max_consecutive_losses": _as_float(
            _first_present(metrics, ("max_consecutive_losses",)) if metrics else _first_present(item, ("max_consecutive_losses",))
        ),
        "final_equity": _as_float(
            _first_present(metrics, ("final_equity", "equity")) if metrics else _first_present(item, ("final_equity", "equity"))
        ),
        "total_fees": _as_float(
            _first_present(metrics, ("total_fees",)) if metrics else _first_present(item, ("total_fees",))
        ),
        "expectancy_pct": _as_float(
            _first_present(metrics, ("expectancy_pct", "expectancy")) if metrics else _first_present(item, ("expectancy_pct", "expectancy"))
        ),
    }
    tr = normalized_metrics["total_return_pct"]
    br = normalized_metrics["benchmark_return_pct"]
    if tr is not None and br is not None:
        normalized_metrics["excess_return_pct"] = round(tr - br, 2)
    else:
        normalized_metrics["excess_return_pct"] = None
    normalized_params = {
        "fast_window": _first_present(best_params, ("fast_window", "fast")) or item.get("fast_window"),
        "slow_window": _first_present(best_params, ("slow_window", "slow")) or item.get("slow_window"),
    }
    explicit_success = _as_bool(_first_present(item, ("success", "ok", "completed")))
    success = explicit_success is True or str(item.get("status", "")).lower() in {
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
        "data": {
            "source": data.get("source") or assumptions.get("data_source") or payload.get("data_source") or "N/A",
            "requested_start_date": data.get("requested_start_date") or payload.get("start_date") or "N/A",
            "requested_end_date": data.get("requested_end_date") or payload.get("end_date") or "N/A",
            "actual_start_date": data.get("actual_start_date") or data.get("start_date") or "N/A",
            "actual_end_date": data.get("actual_end_date") or data.get("end_date") or "N/A",
            "sample_count": _first_non_empty(data.get("sample_count"), normalized_metrics.get("sample_count")),
            "price_field": assumptions.get("price_field") or data.get("price_field") or "close",
        },
        "backtest_context": {
            "strategy": payload.get("strategy") or payload.get("strategy_name") or item.get("strategy") or "MA crossover",
            "strategy_rule": assumptions.get("strategy_rule") or payload.get("strategy_rule") or item.get("strategy_rule") or "",
            "frequency": assumptions.get("frequency") or item.get("frequency") or "daily",
            "initial_cash": assumptions.get("initial_cash"),
            "position_sizing": assumptions.get("position_sizing") or "",
            "fees": assumptions.get("fees"),
            "slippage": assumptions.get("slippage"),
            "execution_price": assumptions.get("execution_price") or "close",
            "risk_free_rate": assumptions.get("risk_free_rate"),
            "benchmark": {
                "name": benchmark.get("name") or item.get("benchmark_name") or "N/A",
                "description": benchmark.get("description") or item.get("benchmark_description") or "",
            },
        },
        "assessment": {
            "risk_level": risk_level,
            "is_effective": effective,
            "conclusion": assessment.get("conclusion") or item.get("conclusion") or "",
        },
        "raw": item,
    }


def _required_sample_count(item: Dict[str, Any]) -> Optional[int]:
    best_params = item.get("best_params") if isinstance(item.get("best_params"), dict) else {}
    slow_window = _as_float(_first_present(best_params, ("slow_window", "slow")))
    return int(slow_window) if slow_window is not None else None


def _is_insufficient_quant_item(item: Dict[str, Any]) -> bool:
    status = str(item.get("status") or "").lower()
    if status in {"insufficient_data", "no_data", "empty_data", "data_unavailable"}:
        return True

    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    sample_count = _as_float(_first_non_empty(data.get("sample_count"), metrics.get("sample_count")))
    required_count = _required_sample_count(item)
    return sample_count is not None and required_count is not None and sample_count < required_count


def _build_data_issue_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issue_items = []
    for item in items:
        if item.get("success") and not _is_insufficient_quant_item(item):
            continue
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        assessment = item.get("assessment") if isinstance(item.get("assessment"), dict) else {}
        issue_items.append({
            "symbol": item.get("symbol") or item.get("code") or "N/A",
            "status": item.get("status") or "failed",
            "sample_count": _first_non_empty(data.get("sample_count"), metrics.get("sample_count"), 0),
            "required_sample_count": _required_sample_count(item),
            "conclusion": assessment.get("conclusion") or "回测数据不可用",
        })
    return issue_items


def get_quant_summary_by_code(code: str, summary_path: str) -> Optional[Dict[str, Any]]:
    payload = _load_json(summary_path)
    if not payload:
        return None
    for item in _extract_items(payload):
        if _matches_code(item, code):
            return _normalize_quant_item(item, payload)
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
    data = summary.get("data") or {}
    context = summary.get("backtest_context") or {}
    benchmark = context.get("benchmark") or {}

    risk_level = str(assessment.get("risk_level", "")).lower()
    is_effective = bool(assessment.get("is_effective"))
    sharpe = metrics.get("sharpe_ratio")
    total_return = metrics.get("total_return_pct")
    benchmark_return = metrics.get("benchmark_return_pct")
    max_dd = metrics.get("max_drawdown_pct")
    success = bool(summary.get("success"))
    status = str(summary.get("status") or "").lower()

    excess_return = metrics.get("excess_return_pct")
    trade_count = metrics.get("trade_count")
    guardrail = ""
    if not success or _is_insufficient_quant_item(summary):
        sample_count = _first_non_empty(data.get("sample_count"), metrics.get("sample_count"), "N/A")
        required_count = _required_sample_count(summary) or "N/A"
        guardrail = f"""
⚠️ **回测数据约束（必须遵守）**：该标的回测状态为 {status or 'failed'}，实际样本 {sample_count} 根K线，最低需求 {required_count} 根。
这表示当前回测没有形成可用验证结论，不能把它解读为策略亏损或策略失效；分析建议应提示数据缺失/样本不足，并以行情、基本面和风控信号为主。
"""
    elif risk_level == "high" or (max_dd is not None and max_dd < -30):
        guardrail = f"""
🚨 **回测硬约束（必须遵守）**：该标的风险评级为 HIGH，最大回撤 {max_dd}%，策略在回测区间亏钱或严重跑输。
你必须在分析结论中明确指出该标的高风险特征，操作建议必须偏向保守（减持、观望、严格止损），禁止给出激进买入或无止损买入建议。
"""
    elif not is_effective or (sharpe is not None and sharpe <= 0):
        guardrail = f"""
⚠️ **回测约束（必须遵守）**：该标的量化策略未生效（夏普 {sharpe}），回测显示均线策略无法稳定盈利。
你必须在分析中提及回测结论，操作建议应保持谨慎，避免乐观追高。
"""
    elif total_return is not None and benchmark_return is not None and total_return < benchmark_return:
        guardrail = f"""
⚠️ 该标的策略收益 {total_return}% 跑输基准 {benchmark_return}%，请在分析中提示策略层面的局限性。
"""

    low_trade_warning = ""
    if trade_count is not None and trade_count < 5:
        low_trade_warning = f"\n⚠️ 交易次数仅 {trade_count} 笔，统计显著性不足，胜率/夏普等指标参考价值有限。"

    return f"""

## 📈 量化回测参考

### ⚙️ 回测配置
| 项目 | 值 |
|------|-----|
| 标的 | {summary.get('symbol', 'N/A')} |
| 回测区间 | {data.get('requested_start_date', 'N/A')} ~ {data.get('requested_end_date', 'N/A')} |
| 实际样本 | {data.get('actual_start_date', 'N/A')} ~ {data.get('actual_end_date', 'N/A')}（{data.get('sample_count', 'N/A')} 根K线） |
| 策略规则 | {context.get('strategy_rule', 'N/A')} |
| 数据源/频率 | {data.get('source', 'N/A')} / {context.get('frequency', 'N/A')} |
| 执行价格 | {context.get('execution_price', 'N/A')} |
| 初始资金 | {context.get('initial_cash', 'N/A')} |
| 手续费/滑点 | {context.get('fees', 'N/A')} / {context.get('slippage', 'N/A')} |
| 基准 | {benchmark.get('name', 'N/A')} — {benchmark.get('description', '')} |

### 📊 收益指标
| 指标 | 值 | 说明 |
|------|-----|------|
| 策略总收益 | {total_return if total_return is not None else 'N/A'}% | 策略择时后的绝对收益 |
| 基准收益 | {benchmark_return if benchmark_return is not None else 'N/A'}% | 买入持有同期收益 |
| 超额收益 | {excess_return if excess_return is not None else 'N/A'}pp | 正数=跑赢基准，负数=跑输基准 |
| 年化收益 | {metrics.get('annualized_return_pct', 'N/A')}% | 便于跨周期比较 |
| 最终权益 | {metrics.get('final_equity', 'N/A')} | 含资金曲线终点 |

### 🛡️ 风险指标
| 指标 | 值 | 说明 |
|------|-----|------|
| 最大回撤 | {max_dd if max_dd is not None else 'N/A'}% | 权益从峰到谷的最大跌幅 |
| 年化波动率 | {metrics.get('volatility_pct', 'N/A')}% | 策略收益的年化标准差 |
| 夏普比率 | {sharpe if sharpe is not None else 'N/A'} | 单位风险超额回报（>1可接受，>2优秀） |
| 卡尔玛比率 | {metrics.get('calmar_ratio', 'N/A')} | 年化收益/最大回撤（趋势策略核心指标） |
| 盈亏比(Profit Factor) | {metrics.get('profit_factor', 'N/A')} | 总盈利/总亏损（>1盈利，>2稳健） |
| 最大连胜 | {metrics.get('max_consecutive_wins', 'N/A')} | 最多连续盈利次数 |
| 最大连败 | {metrics.get('max_consecutive_losses', 'N/A')} | 最多连续亏损次数（影响心理/资金压力） |

### 🔄 交易统计
| 指标 | 值 | 说明 |
|------|-----|------|
| 总交易次数 | {trade_count if trade_count is not None else 'N/A'} | 完整开平仓轮次 |
| 平均持仓天数 | {metrics.get('avg_holding_days', 'N/A')}天 | 策略平均持有周期 |
| 胜率 | {metrics.get('win_rate_pct', 'N/A')}% | 盈利交易占比 |
| 总手续费 | {metrics.get('total_fees', 'N/A')} | ⚠️ 当前为0，实际需加成本 |

### 🏷️ 综合评价
- 风险等级：**{risk_level.upper() if risk_level else 'N/A'}**
- 策略生效：{'✅ 是' if is_effective else '❌ 否'}
{low_trade_warning}
{guardrail}"""



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
    wanted_codes = {code for code in wanted_codes if code}
    raw_items = _extract_items(payload)
    matched_raw_items = [
        item
        for item in raw_items
        if not wanted_codes or any(_matches_code(item, code) for code in wanted_codes)
    ]
    items = [
        _normalize_quant_item(item, payload)
        for item in matched_raw_items
    ]
    items = [item for item in items if item.get("symbol")]
    success_count = sum(1 for item in items if item.get("success"))
    effective_count = sum(1 for item in items if (item.get("assessment") or {}).get("is_effective"))
    failed_count = max(len(items) - success_count, 0)
    available_symbols = sorted({_symbol_key(item.get("symbol") or item.get("code")) for item in raw_items})
    unmatched_codes = sorted(code for code in wanted_codes if not any(_matches_code(item, code) for item in raw_items))
    data_issue_items = _build_data_issue_items(items)
    insufficient_count = sum(1 for item in items if _is_insufficient_quant_item(item))
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
        recommended_action = "请重新生成本次股票池的回测摘要，避免复用旧的 stock_pool_backtest_summary.json。"
    elif success_count == 0 and insufficient_count == len(items):
        status = "no_valid_data"
        conclusion = "量化回测已命中本次报告股票，但行情数据缺失或样本不足，未形成可用验证结论"
        recommended_action = "请检查 QUANT_BACKTEST_DATA_SOURCE、回测日期区间和股票代码格式，修复取数后重新生成摘要。"
    elif success_count == 0:
        status = "failed"
        conclusion = "量化回测已命中本次报告股票，但全部执行失败，未形成可用验证结论"
        recommended_action = "请查看回测日志中的异常原因，修复依赖、权限或数据源后重新生成摘要。"
    elif effective_count > 0:
        status = "effective"
        conclusion = f"量化回测生效：{effective_count}/{len(items)} 只股票通过策略验证"
        recommended_action = ""
    else:
        status = "ineffective"
        conclusion = "量化回测未形成有效验证结论"
        recommended_action = "请把该结果仅作为策略参考，不要将其单独作为买卖依据。"

    return {
        "quant_backtests": {
            "enabled": True,
            "status": status,
            "effective": effective_count > 0,
            "conclusion": conclusion,
            "strategy": payload.get("strategy") or payload.get("strategy_name") or "MA crossover",
            "strategy_rule": payload.get("strategy_rule") or (payload.get("assumptions") or {}).get("strategy_rule") or "",
            "start_date": payload.get("start_date") or payload.get("from") or "N/A",
            "end_date": payload.get("end_date") or payload.get("to") or "N/A",
            "data_source": payload.get("data_source") or (payload.get("assumptions") or {}).get("data_source") or "N/A",
            "benchmark_name": (payload.get("benchmark") or {}).get("name", "N/A"),
            "assumptions": payload.get("assumptions") or {},
            "total": len(items),
            "success": success_count,
            "failed": failed_count,
            "insufficient": insufficient_count,
            "items": items,
            "top_by_sharpe": top_by_sharpe,
            "high_risk_items": high_risk_items,
            "data_issue_items": data_issue_items,
            "requested_codes": sorted(wanted_codes),
            "available_symbols": available_symbols,
            "unmatched_codes": unmatched_codes,
            "recommended_action": recommended_action,
            "source_file": str(source_path),
        }
    }
