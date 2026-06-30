# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.services.quant_context_service import (
    format_quant_summary_for_prompt,
    get_quant_summary_by_code,
    load_quant_backtest_context,
)


def _write_summary(payload):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    with handle:
        json.dump(payload, handle, ensure_ascii=False)
    return handle.name


def _make_config(summary_path: str) -> Config:
    return Config(
        stock_list=[],
        quant_backtest_enabled=True,
        quant_backtest_summary_path=summary_path,
    )


class TestQuantContextService(unittest.TestCase):
    def tearDown(self):
        path = getattr(self, "_summary_path", None)
        if path and os.path.exists(path):
            os.unlink(path)

    def _patch_config(self, payload):
        self._summary_path = _write_summary(payload)
        return mock.patch("src.services.quant_context_service.get_config", return_value=_make_config(self._summary_path))

    def test_load_context_marks_missing_report_stock_as_no_match(self):
        payload = {
            "strategy": "MA Crossover (5, 20)",
            "data_source": "akshare",
            "items": [
                {
                    "symbol": "000001.SZ",
                    "success": True,
                    "metrics": {"total_return_pct": 1, "benchmark_return_pct": 0, "sharpe_ratio": 1},
                    "assessment": {"is_effective": True},
                }
            ],
        }

        with self._patch_config(payload):
            context = load_quant_backtest_context([SimpleNamespace(code="600118")])["quant_backtests"]

        self.assertEqual(context["status"], "no_match")
        self.assertEqual(context["unmatched_codes"], ["600118"])
        self.assertIn("重新生成", context["recommended_action"])

    def test_load_context_marks_matched_insufficient_data_as_no_valid_data(self):
        payload = {
            "strategy": "MA Crossover (5, 20)",
            "data_source": "akshare",
            "items": [
                {
                    "symbol": "600118.SH",
                    "success": False,
                    "status": "insufficient_data",
                    "metrics": {},
                    "data": {"sample_count": 0},
                    "best_params": {"fast_window": 5, "slow_window": 20},
                    "assessment": {"is_effective": False, "conclusion": "数据不足，无法回测"},
                }
            ],
        }

        with self._patch_config(payload):
            context = load_quant_backtest_context([SimpleNamespace(code="600118")])["quant_backtests"]

        self.assertEqual(context["status"], "no_valid_data")
        self.assertEqual(context["insufficient"], 1)
        self.assertEqual(context["data_issue_items"][0]["sample_count"], 0)
        self.assertIn("行情数据缺失或样本不足", context["conclusion"])

    def test_prompt_treats_insufficient_data_as_unusable_not_strategy_failure(self):
        payload = {
            "strategy": "MA Crossover (5, 20)",
            "data_source": "akshare",
            "items": [
                {
                    "symbol": "600118.SH",
                    "success": False,
                    "status": "insufficient_data",
                    "metrics": {},
                    "data": {"sample_count": 0},
                    "best_params": {"fast_window": 5, "slow_window": 20},
                    "assessment": {"is_effective": False, "conclusion": "数据不足，无法回测"},
                }
            ],
        }
        self._summary_path = _write_summary(payload)

        summary = get_quant_summary_by_code("600118", self._summary_path)
        prompt = format_quant_summary_for_prompt(summary)

        self.assertIn("不能把它解读为策略亏损或策略失效", prompt)
        self.assertNotIn("均线策略无法稳定盈利", prompt)


if __name__ == "__main__":
    unittest.main()
