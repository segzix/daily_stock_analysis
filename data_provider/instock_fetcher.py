# -*- coding: utf-8 -*-
"""
InStock chip distribution fetcher.

Uses the CYQ algorithm (ported from InStock) to compute chip distribution
from K-line data. Can be used as a fallback when external chip APIs (akshare/tushare)
are rate-limited or unavailable.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from .base import BaseFetcher, DataFetchError, normalize_stock_code
from .realtime_types import ChipDistribution, safe_float
from .instock_cyq import compute_chip_distribution, DEFAULT_CYQ_DAYS

logger = logging.getLogger(__name__)


def _is_hk(stock_code: str) -> bool:
    """Check if code is a Hong Kong stock."""
    c = (stock_code or "").strip().upper()
    return c.startswith("HK")


def _is_us(stock_code: str) -> bool:
    """Check if code is a US stock."""
    c = (stock_code or "").strip()
    return not c.isdigit() and not c.startswith("HK")


def _is_etf(stock_code: str) -> bool:
    """Check if code is an ETF."""
    c = (stock_code or "").strip()
    return c.startswith("51") or c.startswith("15") or c.startswith("58") or c.startswith("56")


class InstockFetcher(BaseFetcher):
    """
    Fetcher that computes chip distribution using the InStock CYQ algorithm.

    Uses akshare's stock_zh_a_hist() for K-line data (with turnover rate),
    then runs the CYQ algorithm locally.
    """

    name: str = "InstockFetcher"
    priority: int = 10  # Lower than primary fetchers for daily data, but used for chip distribution

    # ------------------------------------------------------------------
    # Required BaseFetcher abstract methods (stubs — not used for chip)
    # ------------------------------------------------------------------

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Stub — InstockFetcher is intended for chip distribution, not daily data."""
        raise NotImplementedError("InstockFetcher._fetch_raw_data is not implemented")

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """Stub — InstockFetcher is intended for chip distribution, not daily data."""
        raise NotImplementedError("InstockFetcher._normalize_data is not implemented")

    # ------------------------------------------------------------------
    # Chip Distribution via CYQ
    # ------------------------------------------------------------------

    def get_chip_distribution(self, stock_code: str) -> Optional[ChipDistribution]:
        """
        Compute chip distribution using InStock's CYQ algorithm.

        Strategy:
        1. Get K-line data with turnover (try baostock first, then akshare)
        2. Run CYQ computation
        3. Return ChipDistribution object

        Args:
            stock_code: Stock code (6-digit for A-shares)

        Returns:
            ChipDistribution or None if computation fails.
        """
        stock_code = normalize_stock_code(stock_code)

        if _is_us(stock_code) or _is_hk(stock_code) or _is_etf(stock_code):
            logger.debug(f"[InStock CYQ] {stock_code} 不支持（美股/港股/ETF），跳过")
            return None

        # Get ~400 calendar days of K-line data to ensure enough trading days
        from datetime import timedelta

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

        kdata = self._fetch_klines_with_turnover(stock_code, start_date, end_date)

        if kdata is None or kdata.empty:
            logger.warning(f"[InStock CYQ] {stock_code} K线+换手率数据为空")
            return None

        # Run CYQ computation (crange=1 for latest chip distribution)
        result = compute_chip_distribution(kdata, crange=1)

        if result is None:
            logger.warning(f"[InStock CYQ] {stock_code} 筹码分布计算失败")
            return None

        chip = ChipDistribution(
            code=stock_code,
            date=str(result.date),
            source="instock_cyq",
            profit_ratio=round(result.benefit_part, 4),
            avg_cost=safe_float(result.avg_cost),
            cost_90_low=safe_float(result.cost_90_low),
            cost_90_high=safe_float(result.cost_90_high),
            concentration_90=safe_float(result.concentration_90),
            cost_70_low=safe_float(result.cost_70_low),
            cost_70_high=safe_float(result.cost_70_high),
            concentration_70=safe_float(result.concentration_70),
        )

        logger.info(
            f"[InStock CYQ] {stock_code} 成功: "
            f"日期={chip.date}, 获利比例={chip.profit_ratio:.1%}, "
            f"平均成本={chip.avg_cost}, 90%集中度={chip.concentration_90:.2%}"
        )
        return chip

    @staticmethod
    def _fetch_klines_with_turnover(
        stock_code: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch K-line data with turnover rate column.

        Tries baostock first (independent of East Money), then akshare as fallback.
        Returns a DataFrame with columns: date, open, close, high, low, turnover.
        """
        kdata = None

        # --- Try baostock first (not blocked by East Money) ---
        try:
            import baostock as bs

            lg = bs.login()
            if lg.error_code != "0":
                logger.debug(f"[InStock CYQ] baostock login failed: {lg.error_msg}")
            else:
                bs_code = f"sh.{stock_code}" if stock_code.startswith(("6", "9")) else f"sz.{stock_code}"
                rs = bs.query_history_k_data_plus(
                    code=bs_code,
                    fields="date,open,high,low,close,volume,amount,turn",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="2",
                )
                if rs.error_code == "0":
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    if rows:
                        columns = ["date", "open", "high", "low", "close", "volume", "amount", "turn"]
                        kdata_df = pd.DataFrame(rows, columns=columns)
                        kdata_df[["open", "high", "low", "close", "turn"]] = kdata_df[
                            ["open", "high", "low", "close", "turn"]
                        ].astype(float)
                        kdata = kdata_df.rename(columns={"turn": "turnover"})
                        logger.info(
                            f"[InStock CYQ] baostock {stock_code}: {len(kdata)} bars, "
                            f"{start_date}~{end_date}"
                        )
                bs.logout()
        except Exception as e:
            logger.debug(f"[InStock CYQ] baostock failed: {e}")

        # --- Fallback: akshare (East Money, may be blocked) ---
        if kdata is None:
            try:
                import akshare as ak

                em_start = start_date.replace("-", "")
                em_end = end_date.replace("-", "")
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=em_start,
                    end_date=em_end,
                    adjust="qfq",
                )
                if df is not None and not df.empty:
                    col_map = {
                        "日期": "date",
                        "开盘": "open",
                        "收盘": "close",
                        "最高": "high",
                        "最低": "low",
                        "换手率": "turnover",
                    }
                    available = {k: v for k, v in col_map.items() if k in df.columns}
                    kdata = df.rename(columns=available)
                    logger.info(f"[InStock CYQ] akshare {stock_code}: {len(kdata)} bars")
            except ImportError:
                logger.debug("[InStock CYQ] akshare not installed")
            except Exception as e:
                logger.debug(f"[InStock CYQ] akshare failed: {e}")

        if kdata is not None and not kdata.empty:
            required = {"date", "open", "close", "high", "low", "turnover"}
            if required.issubset(set(kdata.columns)):
                kdata = kdata[list(required | {"volume", "amount"} & set(kdata.columns))]
                return kdata
            else:
                logger.warning(
                    f"[InStock CYQ] {stock_code} 缺少必要列: {required - set(kdata.columns)}"
                )

        return None
