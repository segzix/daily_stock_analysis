# -*- coding: utf-8 -*-
"""Quant strategy backtest service using vectorbt with akshare data source.

Runs MA crossover backtest on each stock in the configured stock list and
produces a summary JSON consumed by quant_context_service for prompt/report injection.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INITIAL_CASH = 100000.0
FEES = 0.0
SLIPPAGE = 0.0
FREQUENCY = "D"
BENCHMARK_NAME = "Buy-and-hold close-to-close return"
BENCHMARK_DESCRIPTION = "Uses the first and last available close price in the same local data window."


def _is_etf_code(stock_code: str) -> bool:
    code_clean = (stock_code or "").upper().split(".")[0]
    if len(code_clean) < 5:
        return False
    first_two = code_clean[:2]
    first_three = code_clean[:3]
    return first_two in ("51", "52", "56", "58") or first_three in (
        "159", "160", "161", "162", "163", "164",
        "165", "166", "167", "168", "169", "180",
        "184", "186", "188",
    )


def _date_with_dash(value: str) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _normalize_kline_frame(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    normalized = df.copy()
    column_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "money": "amount",
    }
    normalized = normalized.rename(columns={key: value for key, value in column_map.items() if key in normalized.columns})

    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"])
        normalized = normalized.set_index("date")
    elif not isinstance(normalized.index, pd.DatetimeIndex):
        normalized.index = pd.to_datetime(normalized.index)

    if "close" in normalized.columns:
        for col in ("open", "high", "low"):
            if col not in normalized.columns:
                normalized[col] = normalized["close"]
    if "volume" not in normalized.columns:
        normalized["volume"] = 0

    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in normalized.columns]
    if missing_cols:
        logger.warning("Missing required backtest columns: %s", missing_cols)
        return None

    normalized = normalized[required_cols].sort_index()
    for col in required_cols:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    normalized = normalized.dropna(subset=["close"])
    return normalized if not normalized.empty else None


def _to_joinquant_code(stock_code: str) -> str:
    code = (stock_code or "").strip().upper()
    if "." in code:
        digits, market = code.split(".", 1)
    else:
        digits = code
        market = "SH" if digits.startswith(("5", "6", "9")) else "SZ"

    if market in {"SH", "SS", "XSHG"}:
        return f"{digits}.XSHG"
    if market in {"SZ", "XSHE"}:
        return f"{digits}.XSHE"
    if market in {"BJ", "XBEI"}:
        return f"{digits}.XBEI"
    return code


def _fetch_daily_kline_with_fetcher(
    stock_code: str,
    start_date: str,
    end_date: str,
    source: str,
) -> Optional[pd.DataFrame]:
    code_clean = (stock_code or "").strip().upper().split(".")[0]
    start = _date_with_dash(start_date)
    end = _date_with_dash(end_date)

    if source == "efinance":
        from data_provider.efinance_fetcher import EfinanceFetcher

        return _normalize_kline_frame(EfinanceFetcher().get_daily_data(code_clean, start_date=start, end_date=end))

    from data_provider.akshare_fetcher import AkshareFetcher

    return _normalize_kline_frame(AkshareFetcher(sleep_min=0.0, sleep_max=0.0).get_daily_data(
        code_clean,
        start_date=start,
        end_date=end,
    ))


def _fetch_daily_kline_from_joinquant(stock_code: str, start_date: str, end_date: str, config: Any) -> Optional[pd.DataFrame]:
    try:
        import jqdatasdk as jq

        username = getattr(config, "joinquant_username", None)
        password = getattr(config, "joinquant_password", None)
        if username and password:
            jq.auth(username, password)

        df = jq.get_price(
            _to_joinquant_code(stock_code),
            start_date=_date_with_dash(start_date),
            end_date=_date_with_dash(end_date),
            frequency="daily",
            fields=["open", "close", "high", "low", "volume"],
            skip_paused=False,
            fq="pre",
        )
        return _normalize_kline_frame(df)
    except Exception as e:
        logger.warning("JoinQuant fetch failed for %s: %s", stock_code, e)
        return None


def _fetch_daily_kline(
    stock_code: str,
    start_date: str,
    end_date: str,
    data_source: str,
    config: Any,
) -> Optional[pd.DataFrame]:
    source = (data_source or "akshare").strip().lower()
    if source in {"akshare", "akshare_em", "akshare_sina", "tencent", "akshare_qq"}:
        return _fetch_daily_kline_with_fetcher(stock_code, start_date, end_date, "akshare")
    if source == "efinance":
        return _fetch_daily_kline_with_fetcher(stock_code, start_date, end_date, "efinance")
    if source == "joinquant":
        return _fetch_daily_kline_from_joinquant(stock_code, start_date, end_date, config)

    logger.warning("Unsupported QUANT_BACKTEST_DATA_SOURCE=%s, falling back to akshare", data_source)
    return _fetch_daily_kline_with_fetcher(stock_code, start_date, end_date, "akshare")


def _safe_portfolio_metric(method: Any, multiplier: float = 1.0) -> float:
    try:
        value = method() if callable(method) else method
        if value is None or pd.isna(value):
            return 0.0
        return round(float(value) * multiplier, 2)
    except Exception:
        return 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (pd.Series,)):
            value = value.iloc[0]
        v = float(value)
        if pd.isna(v):
            return default
        return round(v, 4)
    except Exception:
        return default


def _max_consecutive(pnl_series: Any, positive: bool = True) -> int:
    try:
        if pnl_series is None or len(pnl_series) == 0:
            return 0
        max_streak = 0
        current_streak = 0
        for val in pnl_series:
            if isinstance(val, (int, float)):
                hit = val > 0 if positive else val < 0
            else:
                hit = (float(val) > 0) if positive else (float(val) < 0)
            if hit:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak
    except Exception:
        return 0


def _build_backtest_assumptions(
    data_source: str, configured_data_source: str, fast_window: int, slow_window: int
) -> Dict[str, Any]:
    return {
        "data_source": data_source,
        "configured_data_source": configured_data_source,
        "price_field": "close",
        "frequency": "daily",
        "strategy_rule": (
            f"Long-only MA crossover: enter when MA{fast_window} crosses above MA{slow_window}; "
            f"exit when MA{fast_window} crosses below MA{slow_window}."
        ),
        "initial_cash": INITIAL_CASH,
        "position_sizing": "vectorbt default all-available-cash signal sizing",
        "fees": FEES,
        "slippage": SLIPPAGE,
        "execution_price": "close",
        "benchmark": {
            "name": BENCHMARK_NAME,
            "description": BENCHMARK_DESCRIPTION,
        },
        "risk_free_rate": 0.0,
    }


def _run_ma_crossover_backtest(
    df: pd.DataFrame, fast_window: int = 5, slow_window: int = 20
) -> Optional[Dict[str, Any]]:
    try:
        import vectorbt as vbt

        close = df["close"]

        fast_ma = vbt.MA.run(close, window=fast_window)
        slow_ma = vbt.MA.run(close, window=slow_window)

        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            freq=FREQUENCY,
            init_cash=INITIAL_CASH,
            fees=FEES,
            slippage=SLIPPAGE,
        )

        trades = pf.trades
        benchmark_return_pct = round((float(close.iloc[-1]) / float(close.iloc[0]) - 1.0) * 100, 2)
        annualized_return_pct = _safe_portfolio_metric(getattr(pf, "annualized_return", None), 100)
        volatility_pct = _safe_portfolio_metric(getattr(pf, "annualized_volatility", None), 100)
        data_start_date = close.index.min().strftime("%Y%m%d") if hasattr(close.index.min(), "strftime") else str(close.index.min())
        data_end_date = close.index.max().strftime("%Y%m%d") if hasattr(close.index.max(), "strftime") else str(close.index.max())

        sample_count = int(close.dropna().shape[0])
        base_metrics = {
            "total_return_pct": 0.0,
            "annualized_return_pct": annualized_return_pct,
            "benchmark_return_pct": benchmark_return_pct,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "volatility_pct": volatility_pct,
            "win_rate_pct": 0.0,
            "trade_count": 0,
            "sample_count": sample_count,
            "data_start_date": data_start_date,
            "data_end_date": data_end_date,
            "start_price": round(float(close.iloc[0]), 4),
            "end_price": round(float(close.iloc[-1]), 4),
            "calmar_ratio": 0.0,
            "profit_factor": 0.0,
            "avg_holding_days": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "final_equity": INITIAL_CASH,
            "total_fees": 0.0,
            "expectancy_pct": 0.0,
        }
        if trades is None or (hasattr(trades, "count") and (trades.count() if callable(trades.count) else trades.count) == 0):
            return base_metrics

        total_return_pct = round(float(pf.total_return() * 100), 2)
        max_drawdown_pct = round(float(pf.max_drawdown() * 100), 2)
        sharpe = round(float(pf.sharpe_ratio() or 0.0), 2)

        wr = trades.win_rate() if callable(trades.win_rate) else trades.win_rate
        if isinstance(wr, (pd.Series,)):
            win_rate_pct = round(float(wr.iloc[0]) * 100, 2) if len(wr) > 0 else 0.0
        elif isinstance(wr, (float, np.floating)):
            win_rate_pct = round(float(wr) * 100, 2)
        elif hasattr(wr, "mean"):
            win_rate_pct = round(float(wr.mean()) * 100, 2)
        else:
            win_rate_pct = 0.0

        tc = trades.count() if callable(trades.count) else trades.count
        trade_count = int(tc) if tc is not None else 0

        calmar_ratio = round(annualized_return_pct / abs(max_drawdown_pct), 2) if max_drawdown_pct != 0 else 0.0

        pf_stats = {}
        try:
            stats_result = pf.stats() if callable(pf.stats) else pf.stats
            if isinstance(stats_result, pd.Series):
                pf_stats = {str(k): v for k, v in stats_result.items()}
            elif isinstance(stats_result, dict):
                pf_stats = {str(k): v for k, v in stats_result.items()}
        except Exception:
            pass

        def _stat(keys, default=0.0):
            for key in keys:
                val = pf_stats.get(key)
                if val is not None:
                    return _safe_float(val)
            return default

        profit_factor = _stat(["Profit Factor", "ProfitFactor", "profit_factor"])

        avg_holding_days = 0.0
        try:
            dur_val = _stat(["Avg Holding Period [days]", "Avg Holding Days", "avg_holding_days",
                             "Avg Trade Duration", "Average Trade Duration"])
            if dur_val != 0:
                avg_holding_days = dur_val
            else:
                durations = getattr(trades, "duration", None)
                if durations is not None and hasattr(durations, "mean"):
                    dur_mean = durations.mean()
                    avg_holding_days = round(float(dur_mean.total_seconds() / 86400 if hasattr(dur_mean, "total_seconds") else dur_mean), 1)
        except Exception:
            pass

        max_consecutive_wins = int(_stat(["Max Consecutive Wins", "max_consecutive_wins"]))
        max_consecutive_losses = int(_stat(["Max Consecutive Losses", "max_consecutive_losses"]))

        final_equity = INITIAL_CASH
        try:
            end_val = _stat(["End Value", "EndValue", "final_value"])
            if end_val > 0:
                final_equity = end_val
            else:
                pf_value = getattr(pf, "value", None)
                if pf_value is not None and len(pf_value) > 0:
                    final_equity = _safe_float(float(pf_value[-1]))
        except Exception:
            pass

        total_fees = _stat(["Total Fees", "TotalFees", "total_fees"])

        expectancy_pct = 0.0
        try:
            exp_val = _stat(["Expectancy", "expectancy"])
            if exp_val != 0:
                expectancy_pct = round(exp_val * 100, 2)
        except Exception:
            pass

        return {
            "total_return_pct": total_return_pct,
            "annualized_return_pct": annualized_return_pct,
            "benchmark_return_pct": benchmark_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe,
            "volatility_pct": volatility_pct,
            "win_rate_pct": win_rate_pct,
            "trade_count": trade_count,
            "sample_count": sample_count,
            "data_start_date": data_start_date,
            "data_end_date": data_end_date,
            "start_price": round(float(close.iloc[0]), 4),
            "end_price": round(float(close.iloc[-1]), 4),
            "calmar_ratio": calmar_ratio,
            "profit_factor": profit_factor,
            "avg_holding_days": avg_holding_days,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "final_equity": final_equity,
            "total_fees": total_fees,
            "expectancy_pct": expectancy_pct,
        }
    except Exception as e:
        logger.warning("Backtest failed: %s", e)
        return None


def _assess_risk(
    total_return: float,
    max_drawdown: float,
    sharpe: float,
    benchmark_return: float,
    calmar_ratio: Optional[float] = None,
    profit_factor: Optional[float] = None,
) -> Dict[str, Any]:
    if max_drawdown < -30:
        risk_level = "high"
    elif max_drawdown < -15:
        risk_level = "medium"
    else:
        risk_level = "low"

    is_effective = bool(
        total_return is not None
        and benchmark_return is not None
        and (total_return >= benchmark_return or sharpe > 0)
    )

    conclusion_parts = []
    if total_return is not None:
        conclusion_parts.append(f"策略收益 {total_return:.1f}%")
    if benchmark_return is not None:
        delta = total_return - benchmark_return if total_return is not None else 0
        conclusion_parts.append(f"基准 {benchmark_return:.1f}%({'+' if delta >= 0 else ''}{delta:.1f}pp)")
    if sharpe is not None:
        conclusion_parts.append(f"夏普 {sharpe:.2f}")
    if calmar_ratio is not None and calmar_ratio > 0:
        conclusion_parts.append(f"卡尔玛 {calmar_ratio:.2f}")
    if profit_factor is not None and profit_factor > 0:
        conclusion_parts.append(f"盈亏比 {profit_factor:.2f}")

    return {
        "risk_level": risk_level,
        "is_effective": is_effective,
        "conclusion": "，".join(conclusion_parts),
    }


def run_stock_pool_backtest(stock_list: Optional[List[str]] = None) -> Dict[str, Any]:
    config = get_config()
    if stock_list is None:
        pool_items = config.resolve_quant_stock_pool_items()
    else:
        pool_items = [{"symbol": code, "start_date": None, "end_date": None} for code in stock_list if code]

    fast_window = getattr(config, "quant_backtest_fast_window", 5) or 5
    slow_window = getattr(config, "quant_backtest_slow_window", 20) or 20
    configured_data_source = (getattr(config, "quant_backtest_data_source", "akshare") or "akshare").strip().lower()
    data_source = configured_data_source
    summary_path = str(
        getattr(config, "quant_backtest_summary_path", "reports/stock_pool_backtest_summary.json")
        or "reports/stock_pool_backtest_summary.json"
    )

    default_start_date, default_end_date = config.resolve_quant_backtest_dates()
    assumptions = _build_backtest_assumptions(data_source, configured_data_source, fast_window, slow_window)

    logger.info(
        "Starting stock pool quant backtest: %d stocks, %s to %s, MA(%d,%d)",
        len(pool_items), default_start_date, default_end_date, fast_window, slow_window,
    )

    items: List[Dict[str, Any]] = []
    success_count = 0
    failed_count = 0

    for pool_item in pool_items:
        code = str(pool_item.get("symbol") or "")
        code_clean = code.strip().upper()
        start_date = pool_item.get("start_date") or default_start_date
        end_date = pool_item.get("end_date") or default_end_date
        logger.info("Backtesting %s ...", code_clean)
        try:
            df = _fetch_daily_kline(code_clean, start_date, end_date, data_source, config)
            if df is None or len(df) < slow_window:
                items.append({
                    "symbol": code_clean,
                    "code": code_clean.split(".")[0],
                    "name": "",
                    "status": "insufficient_data",
                    "success": False,
                    "metrics": {},
                    "data": {
                        "source": data_source,
                        "requested_start_date": start_date,
                        "requested_end_date": end_date,
                        "sample_count": len(df) if df is not None else 0,
                    },
                    "best_params": {"fast_window": fast_window, "slow_window": slow_window},
                    "backtest_context": assumptions,
                    "assessment": {
                        "risk_level": "N/A",
                        "is_effective": False,
                        "conclusion": f"数据不足，无法回测（需≥{slow_window}根K线，实际{len(df) if df is not None else 0}）",
                    },
                })
                failed_count += 1
                continue

            metrics = _run_ma_crossover_backtest(df, fast_window=fast_window, slow_window=slow_window)
            if metrics is None:
                items.append({
                    "symbol": code_clean,
                    "code": code_clean.split(".")[0],
                    "name": "",
                    "status": "error",
                    "success": False,
                    "metrics": {},
                    "data": {
                        "source": data_source,
                        "requested_start_date": start_date,
                        "requested_end_date": end_date,
                        "sample_count": len(df) if df is not None else 0,
                    },
                    "best_params": {"fast_window": fast_window, "slow_window": slow_window},
                    "backtest_context": assumptions,
                    "assessment": {"risk_level": "N/A", "is_effective": False, "conclusion": "回测执行异常"},
                })
                failed_count += 1
                continue

            total_return = metrics.get("total_return_pct", 0)
            benchmark_return = metrics.get("benchmark_return_pct", 0)
            max_dd = metrics.get("max_drawdown_pct", 0)
            sharpe = metrics.get("sharpe_ratio", 0)
            win_rate = metrics.get("win_rate_pct", 0)
            trade_count = metrics.get("trade_count", 0)
            sample_count = metrics.get("sample_count", len(df))

            assessment = _assess_risk(
                total_return, max_dd, sharpe, benchmark_return,
                calmar_ratio=metrics.get("calmar_ratio"),
                profit_factor=metrics.get("profit_factor"),
            )
            item = {
                "symbol": code_clean,
                "code": code_clean.split(".")[0],
                "name": "",
                "status": "completed",
                "success": True,
                "metrics": {
                    "total_return_pct": total_return,
                    "benchmark_return_pct": benchmark_return,
                    "max_drawdown_pct": max_dd,
                    "sharpe_ratio": sharpe,
                    "win_rate_pct": win_rate,
                    "trade_count": trade_count,
                    "annualized_return_pct": metrics.get("annualized_return_pct"),
                    "volatility_pct": metrics.get("volatility_pct"),
                    "sample_count": sample_count,
                    "calmar_ratio": metrics.get("calmar_ratio"),
                    "profit_factor": metrics.get("profit_factor"),
                    "avg_holding_days": metrics.get("avg_holding_days"),
                    "max_consecutive_wins": metrics.get("max_consecutive_wins"),
                    "max_consecutive_losses": metrics.get("max_consecutive_losses"),
                    "final_equity": metrics.get("final_equity"),
                    "total_fees": metrics.get("total_fees"),
                    "expectancy_pct": metrics.get("expectancy_pct"),
                },
                "data": {
                    "source": data_source,
                    "requested_start_date": start_date,
                    "requested_end_date": end_date,
                    "actual_start_date": metrics.get("data_start_date"),
                    "actual_end_date": metrics.get("data_end_date"),
                    "sample_count": sample_count,
                    "start_price": metrics.get("start_price"),
                    "end_price": metrics.get("end_price"),
                },
                "best_params": {"fast_window": fast_window, "slow_window": slow_window},
                "backtest_context": assumptions,
                "assessment": assessment,
            }
            items.append(item)
            success_count += 1
            logger.info(
                "  %s: ret %.1f%%, bench %.1f%%, dd %.1f%%, sharpe %.2f, win %.1f%%, trades %d, risk=%s",
                code_clean, total_return, benchmark_return, max_dd, sharpe, win_rate, trade_count, assessment["risk_level"],
            )
        except Exception as e:
            logger.warning("Unexpected error backtesting %s: %s", code_clean, e)
            items.append({
                "symbol": code_clean,
                "code": code_clean.split(".")[0],
                "name": "",
                "status": "error",
                "success": False,
                "metrics": {},
                "data": {
                    "source": data_source,
                    "requested_start_date": start_date,
                    "requested_end_date": end_date,
                    "sample_count": 0,
                },
                "best_params": {"fast_window": fast_window, "slow_window": slow_window},
                "backtest_context": assumptions,
                "assessment": {"risk_level": "N/A", "is_effective": False, "conclusion": str(e)[:120]},
            })
            failed_count += 1

    summary = {
        "strategy": f"MA Crossover ({fast_window}, {slow_window})",
        "strategy_name": "MA Crossover",
        "strategy_rule": assumptions["strategy_rule"],
        "start_date": default_start_date,
        "end_date": default_end_date,
        "data_source": data_source,
        "configured_data_source": configured_data_source,
        "benchmark": assumptions["benchmark"],
        "assumptions": assumptions,
        "generated_at": datetime.now().isoformat(),
        "total": len(items),
        "success": success_count,
        "failed": failed_count,
        "items": items,
    }

    output_path = PROJECT_ROOT / summary_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Quant backtest summary saved to %s (%d success, %d failed)", output_path, success_count, failed_count)
    return summary


def ensure_quant_backtest_summary(
    stock_list: Optional[List[str]] = None, force: bool = False
) -> Optional[Dict[str, Any]]:
    config = get_config()
    enabled = bool(getattr(config, "quant_backtest_enabled", False))
    if not enabled:
        return None

    summary_path = getattr(
        config, "quant_backtest_summary_path", "reports/stock_pool_backtest_summary.json"
    )
    output_path = PROJECT_ROOT / summary_path

    if not force:
        stale_hours = getattr(config, "quant_backtest_stale_hours", 24) or 24
        if output_path.exists():
            age_seconds = datetime.now().timestamp() - output_path.stat().st_mtime
            if age_seconds < stale_hours * 3600:
                # Check that cached summary matches current stock pool
                try:
                    with output_path.open("r", encoding="utf-8") as f:
                        cached = json.load(f)
                    cached_symbols = {
                        (item.get("symbol") or item.get("code", "")).strip().upper()
                        for item in (cached.get("items") or [])
                    }
                    current_pool = config.resolve_quant_stock_pool_items()
                    current_symbols = {
                        (item.get("symbol") or "").strip().upper().split(".")[0]
                        for item in current_pool
                    }
                    if current_symbols and cached_symbols == current_symbols:
                        logger.info(
                            "Quant backtest summary is fresh and matches stock pool (%d min old), skipping",
                            int(age_seconds / 60),
                        )
                        return cached
                    elif current_symbols:
                        logger.info(
                            "Quant backtest summary stale: cached=%s, current=%s, regenerating",
                            sorted(cached_symbols)[:5], sorted(current_symbols)[:5],
                        )
                    else:
                        # No current pool? Use cached
                        logger.info(
                            "Quant backtest summary is fresh (%d min old), skipping (no current pool to compare)",
                            int(age_seconds / 60),
                        )
                        return cached
                except Exception:
                    logger.warning("Failed to validate cached backtest summary, regenerating")

    auto_run = bool(getattr(config, "quant_backtest_auto_run", False))
    if not auto_run and not force:
        logger.info("Quant backtest auto-run disabled and not forced, skipping")
        return None

    return run_stock_pool_backtest(stock_list=stock_list)
