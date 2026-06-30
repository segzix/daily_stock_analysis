# -*- coding: utf-8 -*-
"""
Chip Distribution (CYQ) algorithm ported from InStock (myhhub/stock).

Computes chip distribution from K-line data (must include 'turnover' column).
Original algorithm: instock/core/kline/cyq.py
Reference: https://github.com/myhhub/stock
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns required by the CYQ algorithm
CYQ_REQUIRED_COLUMNS = {"date", "open", "close", "high", "low", "turnover"}

# Default CYQ parameters (matching InStock defaults)
DEFAULT_ACCURACY_FACTOR = 150
DEFAULT_CRANGE = 120
DEFAULT_CYQ_DAYS = 210


class CYQResult:
    """Chip distribution computation result."""

    def __init__(self):
        self.x: List[float] = []  # Chip stack values per price level
        self.y: List[float] = []  # Price levels
        self.benefit_part: float = 0.0  # Profit ratio (0-1)
        self.avg_cost: float = 0.0  # Average cost
        self.concentration_90: float = 0.0  # 90% chip concentration
        self.concentration_70: float = 0.0  # 70% chip concentration
        self.cost_90_low: float = 0.0
        self.cost_90_high: float = 0.0
        self.cost_70_low: float = 0.0
        self.cost_70_high: float = 0.0
        self.boundary: int = 0
        self.date: str = ""
        self.trading_days: int = 0


def compute_chip_distribution(
    kdata: pd.DataFrame,
    accuracy_factor: int = DEFAULT_ACCURACY_FACTOR,
    crange: int = DEFAULT_CRANGE,
    cyq_days: int = DEFAULT_CYQ_DAYS,
    index: Optional[int] = None,
) -> Optional[CYQResult]:
    """
    Compute chip distribution from K-line data.

    Args:
        kdata: DataFrame with columns [date, open, close, high, low, turnover].
               turnover should be in percent (e.g. 5.0 means 5%).
        accuracy_factor: Number of price buckets (default 150).
        crange: Offset from the last bar (default 120).
        cyq_days: Number of trading days to use (default 210).
        index: Index of the reference bar (default: last row).

    Returns:
        CYQResult or None if data is insufficient.

    Reference:
        InStock CYQCalculator: instock/core/kline/cyq.py
    """
    if kdata is None or kdata.empty:
        logger.warning("CYQ: empty K-line data")
        return None

    missing = CYQ_REQUIRED_COLUMNS - set(kdata.columns)
    if missing:
        logger.warning(f"CYQ: missing required columns: {missing}")
        return None

    if index is None:
        index = len(kdata) - 1

    total_bars = len(kdata)
    if total_bars < cyq_days:
        logger.warning(
            f"CYQ: insufficient data: have {total_bars} bars, need at least {cyq_days}"
        )
        return None

    end = index - crange + 1
    start = end - cyq_days
    if start < 0:
        start = 0
        end = min(cyq_days, total_bars)

    # Slice the relevant window
    if end == 0:
        kdata_window = kdata.tail(min(cyq_days, total_bars))
    else:
        kdata_window = kdata.iloc[max(0, start):max(1, end)]

    if kdata_window.empty:
        logger.warning("CYQ: empty window after slicing")
        return None

    # Compute price range
    maxprice = float(kdata_window["high"].max())
    minprice = float(kdata_window["low"].min())
    if maxprice <= minprice:
        logger.warning("CYQ: invalid price range")
        return None

    # Accuracy (price step) — minimum 0.01
    accuracy = max(0.01, (maxprice - minprice) / (accuracy_factor - 1))

    currentprice = float(kdata_window.iloc[-1]["close"])

    # Build price grid
    yrange = [round(minprice + accuracy * i, 2) for i in range(accuracy_factor)]

    # Find boundary index where price crosses current price
    boundary = -1
    for i, p in enumerate(yrange):
        if boundary == -1 and p >= currentprice:
            boundary = i

    # Chip stack array
    xdata = np.zeros(accuracy_factor, dtype=np.float64)

    # Iterate over each trading day
    for _, row in kdata_window.iterrows():
        open_p = float(row["open"])
        close_p = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        turnover_rate = min(1.0, float(row["turnover"]) / 100.0)

        avg = (open_p + close_p + high + low) / 4.0

        H = int((high - minprice) / accuracy)
        L = int((low - minprice) / accuracy + 0.99)
        L = max(0, L)
        H = min(accuracy_factor - 1, H)

        if high == low:
            GPoint_x = float(accuracy_factor - 1)
        else:
            GPoint_x = 2.0 / (high - low)

        GPoint_y = int((avg - minprice) / accuracy)
        GPoint_y = max(0, min(accuracy_factor - 1, GPoint_y))

        # Decay existing chips by turnover rate
        xdata *= (1.0 - turnover_rate)

        if abs(high - low) < 1e-8:
            # Limit-up/down day: rectangle at double triangle area
            xdata[GPoint_y] += GPoint_x * turnover_rate / 2.0
        else:
            for j in range(L, H + 1):
                curprice = minprice + accuracy * j
                if curprice <= avg:
                    if abs(avg - low) < 1e-8:
                        xdata[j] += GPoint_x * turnover_rate
                    else:
                        xdata[j] += ((curprice - low) / (avg - low)) * GPoint_x * turnover_rate
                else:
                    if abs(high - avg) < 1e-8:
                        xdata[j] += GPoint_x * turnover_rate
                    else:
                        xdata[j] += ((high - curprice) / (high - avg)) * GPoint_x * turnover_rate

    total_chips = float(np.sum(xdata))
    if total_chips <= 0:
        logger.warning("CYQ: total chips is zero")
        return None

    # Helper: get price at a given chip threshold
    def _get_cost_by_chip(chip_amt: float) -> float:
        sum_chips = 0.0
        for i in range(accuracy_factor):
            x_val = float(xdata[i])
            if sum_chips + x_val > chip_amt:
                return minprice + i * accuracy
            sum_chips += x_val
        return maxprice

    # Helper: get profit ratio at a given price
    def _get_benefit_part(price: float) -> float:
        below = 0.0
        for i in range(accuracy_factor):
            if price >= minprice + i * accuracy:
                below += float(xdata[i])
        return below / total_chips if total_chips > 0 else 0.0

    # Helper: compute percentile chip range and concentration
    def _compute_percent_chips(percent: float) -> Tuple[float, float, float]:
        if percent <= 0 or percent > 1:
            raise ValueError(f"percent out of range: {percent}")
        ps_low = (1.0 - percent) / 2.0
        ps_high = (1.0 + percent) / 2.0
        pr_low = _get_cost_by_chip(total_chips * ps_low)
        pr_high = _get_cost_by_chip(total_chips * ps_high)
        concentration = 0.0
        if pr_low + pr_high > 0:
            concentration = (pr_high - pr_low) / (pr_low + pr_high)
        return pr_low, pr_high, concentration

    result = CYQResult()
    result.x = xdata.tolist()
    result.y = yrange
    result.boundary = boundary + 1
    result.date = str(kdata_window.iloc[-1]["date"])
    result.trading_days = cyq_days
    result.benefit_part = _get_benefit_part(currentprice)
    result.avg_cost = round(_get_cost_by_chip(total_chips * 0.5), 2)

    cost_90_low, cost_90_high, concentration_90 = _compute_percent_chips(0.9)
    result.cost_90_low = round(cost_90_low, 2)
    result.cost_90_high = round(cost_90_high, 2)
    result.concentration_90 = round(concentration_90, 4)

    cost_70_low, cost_70_high, concentration_70 = _compute_percent_chips(0.7)
    result.cost_70_low = round(cost_70_low, 2)
    result.cost_70_high = round(cost_70_high, 2)
    result.concentration_70 = round(concentration_70, 4)

    logger.info(
        f"CYQ result: date={result.date}, profit_ratio={result.benefit_part:.1%}, "
        f"avg_cost={result.avg_cost}, concentration_90={result.concentration_90:.2%}"
    )
    return result
