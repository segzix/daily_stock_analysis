# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 LLM 调用逻辑（通过 LiteLLM 统一调用 Gemini/Anthropic/OpenAI 等）
2. 结合技术面和消息面生成分析报告
3. 解析 LLM 响应为结构化 AnalysisResult
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import litellm
from json_repair import repair_json
from litellm import Router

from src.agent.llm_adapter import get_thinking_extra_body
from src.config import Config, get_config, get_api_keys_for_model, extra_litellm_params
from src.storage import persist_llm_usage
from src.data.stock_mapping import STOCK_NAME_MAP
from src.schemas.report_schema import AnalysisReportSchema

logger = logging.getLogger(__name__)


def check_content_integrity(result: "AnalysisResult") -> Tuple[bool, List[str]]:
    """
    Check mandatory fields for report content integrity.
    Returns (pass, missing_fields). Module-level for use by pipeline (agent weak mode).
    """
    missing: List[str] = []
    if result.sentiment_score is None:
        missing.append("sentiment_score")
    if not (result.operation_advice or "").strip():
        missing.append("operation_advice")
    if not (result.analysis_summary or "").strip():
        missing.append("analysis_summary")
    dash = result.dashboard or {}
    core = dash.get("core_conclusion") or {}
    if not (core.get("one_sentence") or "").strip():
        missing.append("dashboard.core_conclusion.one_sentence")
    intel = dash.get("intelligence")
    if intel is None or "risk_alerts" not in intel:
        missing.append("dashboard.intelligence.risk_alerts")
    if result.decision_type in ("buy", "hold"):
        battle = dash.get("battle_plan") or {}
        sp = battle.get("sniper_points") or {}
        stop_loss = sp.get("stop_loss")
        if stop_loss is None or (isinstance(stop_loss, str) and not stop_loss.strip()):
            missing.append("dashboard.battle_plan.sniper_points.stop_loss")
    return len(missing) == 0, missing


def apply_placeholder_fill(result: "AnalysisResult", missing_fields: List[str]) -> None:
    """Fill missing mandatory fields with placeholders (in-place). Module-level for pipeline."""
    for field in missing_fields:
        if field == "sentiment_score":
            result.sentiment_score = 50
        elif field == "operation_advice":
            result.operation_advice = result.operation_advice or "待补充"
        elif field == "analysis_summary":
            result.analysis_summary = result.analysis_summary or "待补充"
        elif field == "dashboard.core_conclusion.one_sentence":
            if not result.dashboard:
                result.dashboard = {}
            if "core_conclusion" not in result.dashboard:
                result.dashboard["core_conclusion"] = {}
            result.dashboard["core_conclusion"]["one_sentence"] = (
                result.dashboard["core_conclusion"].get("one_sentence") or "待补充"
            )
        elif field == "dashboard.intelligence.risk_alerts":
            if not result.dashboard:
                result.dashboard = {}
            if "intelligence" not in result.dashboard:
                result.dashboard["intelligence"] = {}
            if "risk_alerts" not in result.dashboard["intelligence"]:
                result.dashboard["intelligence"]["risk_alerts"] = []
        elif field == "dashboard.battle_plan.sniper_points.stop_loss":
            if not result.dashboard:
                result.dashboard = {}
            if "battle_plan" not in result.dashboard:
                result.dashboard["battle_plan"] = {}
            if "sniper_points" not in result.dashboard["battle_plan"]:
                result.dashboard["battle_plan"]["sniper_points"] = {}
            result.dashboard["battle_plan"]["sniper_points"]["stop_loss"] = "待补充"


# ---------- chip_structure fallback (Issue #589) ----------

_CHIP_KEYS: tuple = ("profit_ratio", "avg_cost", "concentration", "chip_health")


def _is_value_placeholder(v: Any) -> bool:
    """True if value is empty or placeholder (N/A, 数据缺失, etc.)."""
    if v is None:
        return True
    if isinstance(v, (int, float)) and v == 0:
        return True
    s = str(v).strip().lower()
    return s in ("", "n/a", "na", "数据缺失", "未知")


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Safely convert to float; return default on failure. Private helper for chip fill."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        try:
            return default if math.isnan(float(v)) else float(v)
        except (ValueError, TypeError):
            return default
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _derive_chip_health(profit_ratio: float, concentration_90: float) -> str:
    """Derive chip_health from profit_ratio and concentration_90."""
    if profit_ratio >= 0.9:
        return "警惕"  # 获利盘极高
    if concentration_90 >= 0.25:
        return "警惕"  # 筹码分散
    if concentration_90 < 0.15 and 0.3 <= profit_ratio < 0.9:
        return "健康"  # 集中且获利比例适中
    return "一般"


def _build_chip_structure_from_data(chip_data: Any) -> Dict[str, Any]:
    """Build chip_structure dict from ChipDistribution or dict."""
    if hasattr(chip_data, "profit_ratio"):
        pr = _safe_float(chip_data.profit_ratio)
        ac = chip_data.avg_cost
        c90 = _safe_float(chip_data.concentration_90)
    else:
        d = chip_data if isinstance(chip_data, dict) else {}
        pr = _safe_float(d.get("profit_ratio"))
        ac = d.get("avg_cost")
        c90 = _safe_float(d.get("concentration_90"))
    chip_health = _derive_chip_health(pr, c90)
    return {
        "profit_ratio": f"{pr:.1%}",
        "avg_cost": ac if (ac is not None and _safe_float(ac) != 0.0) else "N/A",
        "concentration": f"{c90:.2%}",
        "chip_health": chip_health,
    }


def _get_strategy_thresholds(style: str) -> dict:
    """Return trading thresholds based on strategy style."""
    styles = {
        "conservative": {
            "label": "保守",
            "bias_threshold_pct": 5,
            "executable_score": 65,
            "require_bullish": True,
            "position_max_pct": 20,
            "buy_rule": "缩量回踩 MA5/MA10，乖离率 < 2% 最佳",
            "sell_rule": "跌破 MA20 立即止损",
            "trend_requirement": "多头排列（MA5>MA10>MA20）是买入必要条件",
        },
        "moderate": {
            "label": "均衡",
            "bias_threshold_pct": 7,
            "executable_score": 55,
            "require_bullish": False,
            "position_max_pct": 30,
            "buy_rule": "回踩 MA10/MA20 支撑，或放量突破 MA5 且乖离率 < 3%",
            "sell_rule": "跌破 MA20 且量能放大时减仓",
            "trend_requirement": "至少 MA5 > MA20，不要求严格多头排列",
        },
        "aggressive": {
            "label": "进取",
            "bias_threshold_pct": 10,
            "executable_score": 45,
            "require_bullish": False,
            "position_max_pct": 40,
            "buy_rule": "放量突破关键阻力，或回踩震荡区间下沿。可追涨强势股",
            "sell_rule": "跌破止损位（买入价的 3-5%）或 MA20 确认破位",
            "trend_requirement": "不要求均线多头排列，关注价格动能和成交量",
        },
    }
    return styles.get(style, styles["conservative"])


def fill_chip_structure_if_needed(result: "AnalysisResult", chip_data: Any) -> None:
    """When chip_data exists, fill chip_structure placeholder fields from chip_data (in-place)."""
    if not result or not chip_data:
        return
    try:
        if not result.dashboard:
            result.dashboard = {}
        dash = result.dashboard
        # Use `or {}` rather than setdefault so that an explicit `null` from LLM is also replaced
        dp = dash.get("data_perspective") or {}
        dash["data_perspective"] = dp
        cs = dp.get("chip_structure") or {}
        filled = _build_chip_structure_from_data(chip_data)
        # Start from a copy of cs to preserve any extra keys the LLM may have added
        merged = dict(cs)
        for k in _CHIP_KEYS:
            if _is_value_placeholder(merged.get(k)):
                merged[k] = filled[k]
        if merged != cs:
            dp["chip_structure"] = merged
            logger.info("[chip_structure] Filled placeholder chip fields from data source (Issue #589)")
    except Exception as e:
        logger.warning("[chip_structure] Fill failed, skipping: %s", e)


def _fill_data_perspective_from_context(result: "AnalysisResult", context: dict) -> None:
    """Fill data_perspective from analysis context when LLM doesn't generate it."""
    if not result or not context:
        return
    try:
        dash = result.dashboard if result.dashboard else {}
        dp = dash.get("data_perspective") or {}
        if not isinstance(dp, dict):
            dp = {}
        needs_fill = not dp or not any(dp.get(k) for k in ("trend_status", "price_position", "volume_analysis"))
        if not needs_fill and dp.get("chip_structure"):
            return  # Already fully populated by LLM
        if not needs_fill:
            return

        trend = context.get("trend_analysis") or {}
        prices = (context.get("prices") or {}).get("today") or {}
        realtime = context.get("realtime") or {}

        if not dp.get("trend_status"):
            dp["trend_status"] = {
                "ma_alignment": trend.get("status", "未知"),
                "is_bullish": trend.get("is_bullish", False),
                "trend_score": trend.get("score", 0),
            }
        if not dp.get("price_position"):
            current = prices.get("close", "-") or realtime.get("price", "-")
            raw_bias = str(prices.get("bias_ma5", "")).replace("%", "").strip()
            try:
                bias_val = float(raw_bias)
                bias_text = f"{bias_val:.1f}%"
                if bias_val < 3:
                    bias_status = "安全"
                elif bias_val < 5:
                    bias_status = "警戒"
                else:
                    bias_status = "危险"
            except (ValueError, TypeError):
                bias_text = "-"
                bias_status = "未知"
            dp["price_position"] = {
                "current_price": current,
                "ma5": prices.get("ma5", "-"),
                "ma10": prices.get("ma10", "-"),
                "ma20": prices.get("ma20", "-"),
                "bias_ma5": bias_text,
                "bias_status": bias_status,
                "support_level": prices.get("ma20", "-"),
                "resistance_level": prices.get("ma5", "-"),
            }
        if not dp.get("volume_analysis"):
            vr = realtime.get("volume_ratio", "-")
            try:
                vr_val = float(str(vr))
                if vr_val > 1.2:
                    vol_status = "放量"
                elif vr_val < 0.8:
                    vol_status = "缩量"
                else:
                    vol_status = "平量"
            except (ValueError, TypeError):
                vol_status = "平量"
            dp["volume_analysis"] = {
                "volume_ratio": vr,
                "volume_status": vol_status,
                "turnover_rate": realtime.get("turnover_rate", "-"),
                "volume_meaning": realtime.get("volume_ratio_desc", "数据未提供"),
            }
        dash["data_perspective"] = dp
        result.dashboard = dash
    except Exception:
        pass


def mark_chip_structure_missing(result: "AnalysisResult") -> None:
    """Mark chip_structure fields as unavailable when the data source returned no chip data."""
    if not result:
        return
    if not result.dashboard:
        result.dashboard = {}
    data_perspective = result.dashboard.get("data_perspective") or {}
    result.dashboard["data_perspective"] = data_perspective
    data_perspective["chip_structure"] = {
        "profit_ratio": "数据不足",
        "avg_cost": "数据不足",
        "concentration": "数据不足",
        "chip_health": "无法判断",
    }


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager = None
) -> str:
    """
    多来源获取股票中文名称

    获取策略（按优先级）：
    1. 从传入的 context 中获取（realtime 数据）
    2. 从静态映射表 STOCK_NAME_MAP 获取
    3. 从 DataFetcherManager 获取（各数据源）
    4. 返回默认名称（股票+代码）

    Args:
        stock_code: 股票代码
        context: 分析上下文（可选）
        data_manager: DataFetcherManager 实例（可选）

    Returns:
        股票中文名称
    """
    # 1. 从上下文获取（实时行情数据）
    if context:
        # 优先从 stock_name 字段获取
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('股票'):
                return name

        # 其次从 realtime 数据获取
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']

    # 2. 从静态映射表获取
    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]

    # 3. 从数据源获取
    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception as e:
            logger.debug(f"无法初始化 DataFetcherManager: {e}")

    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                # 更新缓存
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as e:
            logger.debug(f"从数据源获取股票名称失败: {e}")

    # 4. 返回默认名称
    return f'股票{stock_code}'


@dataclass
class AnalysisResult:
    """
    AI 分析结果数据类 - 决策仪表盘版

    封装 Gemini 返回的分析结果，包含决策仪表盘和详细分析
    """
    code: str
    name: str

    # ========== 核心指标 ==========
    sentiment_score: int  # 综合评分 0-100 (>70强烈看多, >60看多, 40-60震荡, <40看空)
    trend_prediction: str  # 趋势预测：强烈看多/看多/震荡/看空/强烈看空
    operation_advice: str  # 操作建议：买入/加仓/持有/减仓/卖出/观望
    decision_type: str = "hold"  # 决策类型：buy/hold/sell（用于统计）
    confidence_level: str = "中"  # 置信度：高/中/低

    # ========== 决策仪表盘 (新增) ==========
    dashboard: Optional[Dict[str, Any]] = None  # 完整的决策仪表盘数据

    # ========== 走势分析 ==========
    trend_analysis: str = ""  # 走势形态分析（支撑位、压力位、趋势线等）
    short_term_outlook: str = ""  # 短期展望（1-3日）
    medium_term_outlook: str = ""  # 中期展望（1-2周）

    # ========== 技术面分析 ==========
    technical_analysis: str = ""  # 技术指标综合分析
    ma_analysis: str = ""  # 均线分析（多头/空头排列，金叉/死叉等）
    volume_analysis: str = ""  # 量能分析（放量/缩量，主力动向等）
    pattern_analysis: str = ""  # K线形态分析

    # ========== 基本面分析 ==========
    fundamental_analysis: str = ""  # 基本面综合分析
    sector_position: str = ""  # 板块地位和行业趋势
    company_highlights: str = ""  # 公司亮点/风险点

    # ========== 情绪面/消息面分析 ==========
    news_summary: str = ""  # 近期重要新闻/公告摘要
    market_sentiment: str = ""  # 市场情绪分析
    hot_topics: str = ""  # 相关热点话题

    # ========== 综合分析 ==========
    analysis_summary: str = ""  # 综合分析摘要
    key_points: str = ""  # 核心看点（3-5个要点）
    risk_warning: str = ""  # 风险提示
    buy_reason: str = ""  # 买入/卖出理由

    # ========== 元数据 ==========
    market_snapshot: Optional[Dict[str, Any]] = None  # 当日行情快照（展示用）
    raw_response: Optional[str] = None  # 原始响应（调试用）
    search_performed: bool = False  # 是否执行了联网搜索
    data_sources: str = ""  # 数据来源说明
    success: bool = True
    error_message: Optional[str] = None

    # ========== 管线追踪（每步的成败和来源）==========
    daily_data_source: str = ""       # 日线数据来源（如 AkshareFetcher）
    daily_data_ok: bool = True        # 日线数据是否成功
    realtime_source: str = ""         # 实时行情来源（如 akshare_em）
    realtime_ok: bool = True          # 实时行情是否成功
    chip_source: str = ""             # 筹码分布来源（如 akshare）
    chip_ok: bool = True              # 筹码分布是否成功
    intel_trace: List[Dict[str, Any]] = field(default_factory=list)
    # intel_trace: [{dim, provider, success, results, error}, ...]

    # ========== 价格数据（分析时快照）==========
    current_price: Optional[float] = None  # 分析时的股价
    change_pct: Optional[float] = None     # 分析时的涨跌幅(%)

    # ========== 模型标记（Issue #528）==========
    model_used: Optional[str] = None  # 分析使用的 LLM 模型（完整名，如 gemini/gemini-2.0-flash）

    # ========== 分步分析中间结果（3-Step 模式）==========
    step1_technical: Optional[Dict[str, Any]] = None  # Step1: 技术面趋势分析
    step2_validation: Optional[Dict[str, Any]] = None  # Step2: 筹码+回测验证

    # ========== 历史对比（Report Engine P0）==========
    query_id: Optional[str] = None  # 本次分析 query_id，用于历史对比时排除本次记录

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'decision_type': self.decision_type,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,  # 决策仪表盘数据
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'market_snapshot': self.market_snapshot,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
            'current_price': self.current_price,
            'change_pct': self.change_pct,
            'model_used': self.model_used,
        }

    def get_core_conclusion(self) -> str:
        """获取核心结论（一句话）"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        """获取持仓建议"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        """获取狙击点位"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}

    def get_checklist(self) -> List[str]:
        """获取检查清单"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []

    def get_risk_alerts(self) -> List[str]:
        """获取风险警报"""
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []

    def get_emoji(self) -> str:
        """根据操作建议返回对应 emoji"""
        emoji_map = {
            '买入': '🟢',
            '加仓': '🟢',
            '强烈买入': '💚',
            '持有': '🟡',
            '观望': '⚪',
            '减仓': '🟠',
            '卖出': '🔴',
            '强烈卖出': '❌',
        }
        advice = self.operation_advice or ''
        # Direct match first
        if advice in emoji_map:
            return emoji_map[advice]
        # Handle compound advice like "卖出/观望" — use the first part
        for part in advice.replace('/', '|').split('|'):
            part = part.strip()
            if part in emoji_map:
                return emoji_map[part]
        # Score-based fallback
        score = self.sentiment_score
        if score >= 80:
            return '💚'
        elif score >= 65:
            return '🟢'
        elif score >= 55:
            return '🟡'
        elif score >= 45:
            return '⚪'
        elif score >= 35:
            return '🟠'
        else:
            return '🔴'

    def get_confidence_stars(self) -> str:
        """返回置信度星级"""
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """
    Gemini AI 分析器

    职责：
    1. 调用 Google Gemini API 进行股票分析
    2. 结合预先搜索的新闻和技术面数据生成分析报告
    3. 解析 AI 返回的 JSON 格式结果

    使用方式：
        analyzer = GeminiAnalyzer()
        result = analyzer.analyze(context, news_context)
    """

    def _build_system_prompt(self) -> str:
        """Build the system prompt with strategy-specific thresholds."""
        from src.config import get_config
        style = getattr(get_config(), "strategy_style", "conservative")
        t = _get_strategy_thresholds(style)
        bias_safe = max(1.0, t["bias_threshold_pct"] * 0.4)
        bias_action = (
            "严禁追高！直接判定为观望"
            if style == "conservative"
            else "谨慎介入，仅限轻仓试错"
        )
        replacements = {
            "__STYLE_LABEL__": {
                "conservative": "保守型（严进宽出，宁可错过不买错）",
                "moderate": "均衡型（平衡风险与收益）",
                "aggressive": "进取型（放宽趋势要求，允许追涨）",
            }.get(style, "保守型"),
            "__BUY_RULE__": t["buy_rule"],
            "__BIAS_SAFE__": str(bias_safe),
            "__BIAS_THRESHOLD_PCT__": str(t["bias_threshold_pct"]),
            "__BIAS_ACTION__": bias_action,
            "__TREND_REQUIREMENT__": t["trend_requirement"],
            "__SELL_RULE__": t["sell_rule"],
            "__POSITION_MAX_PCT__": str(t["position_max_pct"]),
        }
        prompt = self.SYSTEM_PROMPT
        for key, value in replacements.items():
            prompt = prompt.replace("{" + key + "}", value)
        return prompt

    # ========================================
    # 系统提示词 - 决策仪表盘 v2.0
    # ========================================
    # 输出格式升级：从简单信号升级为决策仪表盘
    # 核心模块：核心结论 + 数据透视 + 舆情情报 + 作战计划
    # ========================================

    SYSTEM_PROMPT = """你是一位专注于趋势交易的 A 股投资分析师，负责生成专业的【决策仪表盘】分析报告。
当前策略风格：{__STYLE_LABEL__}

## 核心交易理念（必须严格遵守）

### 1. 入场策略
- {__BUY_RULE__}
- 乖离率公式：(现价 - MA5) / MA5 × 100%
- 乖离率 {__BIAS_SAFE__}%：安全区间，可正常操作
- 乖离率 {__BIAS_SAFE__}-{__BIAS_THRESHOLD_PCT__}%：谨慎区间，可轻仓
- 乖离率 > {__BIAS_THRESHOLD_PCT__}%：风险区间。**超过 {__BIAS_THRESHOLD_PCT__}% 时：{__BIAS_ACTION__}**

### 2. 趋势判断
- {__TREND_REQUIREMENT__}
- 均线发散上行优于均线粘合
- 趋势强度判断：看均线间距是否在扩大

### 3. 效率优先（筹码结构）
- 关注筹码集中度：90%集中度 < 15% 表示筹码集中
- 获利比例分析：70-90% 获利盘时需警惕获利回吐
- 平均成本与现价关系：现价高于平均成本 5-15% 为健康

### 4. 卖出规则
- {__SELL_RULE__}
- 建议仓位上限：{__POSITION_MAX_PCT__}%

### 5. 风险排查重点
- 减持公告（股东、高管减持）
- 业绩预亏/大幅下滑
- 监管处罚/立案调查
- 行业政策利空
- 大额解禁

### 6. 估值关注（PE/PB）
- 分析时请关注市盈率（PE）是否合理
- PE 明显偏高时（如远超行业平均或历史均值），需在风险点中说明
- 高成长股可适当容忍较高 PE，但需有业绩支撑

### 7. 强势趋势股
- 强势趋势股（趋势强度高、量能配合）可适当放宽乖离率要求
- 此类股票可轻仓追踪，但仍需设置止损，不盲目追高

## 输出格式：决策仪表盘 JSON

请严格按照以下 JSON 格式输出，这是一个完整的【决策仪表盘】：

```json
{
    "stock_name": "股票中文名称",
    "sentiment_score": 0-100整数,
    "trend_prediction": "强烈看多/看多/震荡/看空/强烈看空",
    "operation_advice": "买入/加仓/持有/减仓/卖出/观望",
    "decision_type": "buy/hold/sell",
    "confidence_level": "高/中/低",

    "dashboard": {
        "core_conclusion": {
            "one_sentence": "一句话核心结论（30字以内，直接告诉用户做什么）",
            "signal_type": "🟢买入信号/🟡持有观望/🔴卖出信号/⚠️风险警告",
            "time_sensitivity": "立即行动/今日内/本周内/不急",
            "position_advice": {
                "no_position": "空仓者建议：具体操作指引",
                "has_position": "持仓者建议：具体操作指引"
            }
        },

        "data_perspective": {
            "trend_status": {
                "ma_alignment": "均线排列状态描述",
                "is_bullish": true/false,
                "trend_score": 0-100
            },
            "price_position": {
                "current_price": 当前价格数值,
                "ma5": MA5数值,
                "ma10": MA10数值,
                "ma20": MA20数值,
                "bias_ma5": 乖离率百分比数值,
                "bias_status": "安全/警戒/危险",
                "support_level": 支撑位价格,
                "resistance_level": 压力位价格
            },
            "volume_analysis": {
                "volume_ratio": 量比数值,
                "volume_status": "放量/缩量/平量",
                "turnover_rate": 换手率百分比,
                "volume_meaning": "量能含义解读（如：缩量回调表示抛压减轻）"
            },
            "chip_structure": {
                "profit_ratio": 获利比例,
                "avg_cost": 平均成本,
                "concentration": 筹码集中度,
                "chip_health": "健康/一般/警惕"
            }
        },

        "intelligence": {
            "latest_news": "【最新消息】近期重要新闻摘要",
            "risk_alerts": ["风险点1：具体描述", "风险点2：具体描述"],
            "positive_catalysts": ["利好1：具体描述", "利好2：具体描述"],
            "earnings_outlook": "业绩预期分析（基于年报预告、业绩快报等）",
            "sentiment_summary": "舆情情绪一句话总结"
        },

        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "理想买入点：XX元（在MA5附近）",
                "secondary_buy": "次优买入点：XX元（在MA10附近）",
                "stop_loss": "止损位：XX元（跌破MA20或X%）",
                "take_profit": "目标位：XX元（前高/整数关口）"
            },
            "position_strategy": {
                "suggested_position": "建议仓位：X成",
                "entry_plan": "分批建仓策略描述",
                "risk_control": "风控策略描述"
            },
            "action_checklist": [
                "✅/⚠️/❌ 检查项1：多头排列",
                "✅/⚠️/❌ 检查项2：乖离率合理（强势趋势可放宽）",
                "✅/⚠️/❌ 检查项3：量能配合",
                "✅/⚠️/❌ 检查项4：无重大利空",
                "✅/⚠️/❌ 检查项5：筹码健康",
                "✅/⚠️/❌ 检查项6：PE估值合理"
            ]
        }
    },

    "analysis_summary": "100字综合分析摘要",
    "key_points": "3-5个核心看点，逗号分隔",
    "risk_warning": "风险提示",
    "buy_reason": "操作理由，引用交易理念",

    "trend_analysis": "走势形态分析",
    "short_term_outlook": "短期1-3日展望",
    "medium_term_outlook": "中期1-2周展望",
    "technical_analysis": "技术面综合分析",
    "ma_analysis": "均线系统分析",
    "volume_analysis": "量能分析",
    "pattern_analysis": "K线形态分析",
    "fundamental_analysis": "基本面分析",
    "sector_position": "板块行业分析",
    "company_highlights": "公司亮点/风险",
    "news_summary": "新闻摘要",
    "market_sentiment": "市场情绪",
    "hot_topics": "相关热点",

    "search_performed": true/false,
    "data_sources": "数据来源说明"
}
```

## 评分标准

### 强烈买入（80-100分）：
- ✅ 多头排列：MA5 > MA10 > MA20
- ✅ 低乖离率：<2%，最佳买点
- ✅ 缩量回调或放量突破
- ✅ 筹码集中健康
- ✅ 消息面有利好催化

### 买入（60-79分）：
- ✅ 多头排列或弱势多头
- ✅ 乖离率 <5%
- ✅ 量能正常
- ⚪ 允许一项次要条件不满足

### 观望（40-59分）：
- ⚠️ 乖离率 >5%（追高风险）
- ⚠️ 均线缠绕趋势不明
- ⚠️ 有风险事件

### 卖出/减仓（0-39分）：
- ❌ 空头排列
- ❌ 跌破MA20
- ❌ 放量下跌
- ❌ 重大利空

## 决策仪表盘核心原则

1. **核心结论先行**：一句话说清该买该卖
2. **分持仓建议**：空仓者和持仓者给不同建议
3. **精确狙击点**：必须给出具体价格，不说模糊的话
4. **检查清单可视化**：用 ✅⚠️❌ 明确显示每项检查结果
5. **风险优先级**：舆情中的风险点要醒目标出"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM Analyzer via LiteLLM.

        Args:
            api_key: Ignored (kept for backward compatibility). Keys are loaded from config.
        """
        self._router = None
        self._litellm_available = False
        self._init_litellm()
        if not self._litellm_available:
            logger.warning("No LLM configured (LITELLM_MODEL / API keys), AI analysis will be unavailable")

    def _has_channel_config(self, config: Config) -> bool:
        """Check if multi-channel config (channels / YAML / legacy model_list) is active."""
        return bool(config.llm_model_list) and not all(
            e.get('model_name', '').startswith('__legacy_') for e in config.llm_model_list
        )

    def _init_litellm(self) -> None:
        """Initialize litellm Router from channels / YAML / legacy keys."""
        config = get_config()
        litellm_model = config.litellm_model
        if not litellm_model:
            logger.warning("Analyzer LLM: LITELLM_MODEL not configured")
            return

        self._litellm_available = True

        # --- Channel / YAML path: build Router from pre-built model_list ---
        if self._has_channel_config(config):
            model_list = config.llm_model_list
            self._router = Router(
                model_list=model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            unique_models = list(dict.fromkeys(
                e['litellm_params']['model'] for e in model_list
            ))
            logger.info(
                f"Analyzer LLM: Router initialized from channels/YAML — "
                f"{len(model_list)} deployment(s), models: {unique_models}"
            )
            return

        # --- Legacy path: build Router for multi-key, or use single key ---
        keys = get_api_keys_for_model(litellm_model, config)

        if len(keys) > 1:
            # Build legacy Router for primary model multi-key load-balancing
            extra_params = extra_litellm_params(litellm_model, config)
            legacy_model_list = [
                {
                    "model_name": litellm_model,
                    "litellm_params": {
                        "model": litellm_model,
                        "api_key": k,
                        **extra_params,
                    },
                }
                for k in keys
            ]
            self._router = Router(
                model_list=legacy_model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            logger.info(
                f"Analyzer LLM: Legacy Router initialized with {len(keys)} keys "
                f"for {litellm_model}"
            )
        elif keys:
            logger.info(f"Analyzer LLM: litellm initialized (model={litellm_model})")
        else:
            logger.info(
                f"Analyzer LLM: litellm initialized (model={litellm_model}, "
                f"API key from environment)"
            )

        # Confirm the model and key being used
        _keys = get_api_keys_for_model(litellm_model, config)
        if _keys:
            _masked = _keys[0][:10] + "…" if len(_keys[0]) > 10 else "***"
            logger.info(f"[LLM路由] 模型: {litellm_model} | API Key: {_masked}")
        else:
            logger.info(f"[LLM路由] 模型: {litellm_model} | API Key: 由环境变量/provider自动解析")

    def is_available(self) -> bool:
        """Check if LiteLLM is properly configured with at least one API key."""
        return self._router is not None or self._litellm_available

    def _call_litellm(self, prompt: str, generation_config: dict) -> Tuple[str, str, Dict[str, Any]]:
        """Call LLM via litellm with fallback across configured models.

        When channels/YAML are configured, every model goes through the Router
        (which handles per-model key selection, load balancing, and retries).
        In legacy mode, the primary model may use the Router while fallback
        models fall back to direct litellm.completion().

        Args:
            prompt: User prompt text.
            generation_config: Dict with optional keys: temperature, max_output_tokens, max_tokens.

        Returns:
            Tuple of (response text, model_used, usage). On success model_used is the full model
            name and usage is a dict with prompt_tokens, completion_tokens, total_tokens.
        """
        config = get_config()
        max_tokens = (
            generation_config.get('max_output_tokens')
            or generation_config.get('max_tokens')
            or 8192
        )
        temperature = generation_config.get('temperature', 0.7)

        models_to_try = [config.litellm_model] + (config.litellm_fallback_models or [])
        models_to_try = [m for m in models_to_try if m]

        use_channel_router = self._has_channel_config(config)

        last_error = None
        for model in models_to_try:
            try:
                model_short = model.split("/")[-1] if "/" in model else model
                call_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self._build_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if getattr(config, "debug_print_prompt", False):
                    _sys = call_kwargs["messages"][0]["content"]
                    _usr = call_kwargs["messages"][1]["content"]
                    logger.warning(
                        "\n========== DEBUG_PROMPT [model=%s] len(system)=%d len(user)=%d ==========\n"
                        "=== SYSTEM ===\n%s\n"
                        "=== USER ===\n%s\n"
                        "========== DEBUG_PROMPT END ==========",
                        model, len(_sys), len(_usr), _sys, _usr,
                    )
                extra = get_thinking_extra_body(model_short)
                if extra:
                    call_kwargs["extra_body"] = extra

                if use_channel_router and self._router:
                    # Channel / YAML path: Router manages key + base_url per model
                    response = self._router.completion(**call_kwargs)
                elif self._router and model == config.litellm_model:
                    # Legacy path: Router only for primary model multi-key
                    response = self._router.completion(**call_kwargs)
                else:
                    # Legacy path: direct call for fallback models
                    keys = get_api_keys_for_model(model, config)
                    if keys:
                        call_kwargs["api_key"] = keys[0]
                    call_kwargs.update(extra_litellm_params(model, config))
                    response = litellm.completion(**call_kwargs)

                if response and response.choices and response.choices[0].message.content:
                    usage: Dict[str, Any] = {}
                    if response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens or 0,
                            "completion_tokens": response.usage.completion_tokens or 0,
                            "total_tokens": response.usage.total_tokens or 0,
                        }
                    return (response.choices[0].message.content, model, usage)
                raise ValueError("LLM returned empty response")

            except Exception as e:
                logger.warning(f"[LiteLLM] {model} failed: {e}")
                last_error = e
                continue

        raise Exception(f"All LLM models failed (tried {len(models_to_try)} model(s)). Last error: {last_error}")

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Public entry point for free-form text generation.

        External callers (e.g. MarketAnalyzer) must use this method instead of
        calling _call_litellm() directly or accessing private attributes such as
        _litellm_available, _router, _model, _use_openai, or _use_anthropic.

        Args:
            prompt:      Text prompt to send to the LLM.
            max_tokens:  Maximum tokens in the response (default 2048).
            temperature: Sampling temperature (default 0.7).

        Returns:
            Response text, or None if the LLM call fails (error is logged).
        """
        try:
            result = self._call_litellm(
                prompt,
                generation_config={"max_tokens": max_tokens, "temperature": temperature},
            )
            if isinstance(result, tuple):
                text, model_used, usage = result
                persist_llm_usage(usage, model_used, call_type="market_review")
                return text
            return result
        except Exception as exc:
            logger.error("[generate_text] LLM call failed: %s", exc)
            return None

    def analyze(
        self, 
        context: Dict[str, Any],
        news_context: Optional[str] = None
    ) -> AnalysisResult:
        """
        分析单只股票
        
        流程：
        1. 格式化输入数据（技术面 + 新闻）
        2. 调用 Gemini API（带重试和模型切换）
        3. 解析 JSON 响应
        4. 返回结构化结果
        
        Args:
            context: 从 storage.get_analysis_context() 获取的上下文数据
            news_context: 预先搜索的新闻内容（可选）
            
        Returns:
            AnalysisResult 对象
        """
        code = context.get('code', 'Unknown')
        config = get_config()
        
        # 请求前增加延时（防止连续请求触发限流）
        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug(f"[LLM] 请求前等待 {request_delay:.1f} 秒...")
            time.sleep(request_delay)
        
        # 优先从上下文获取股票名称（由 main.py 传入）
        name = context.get('stock_name')
        if not name or name.startswith('股票'):
            # 备选：从 realtime 中获取
            if 'realtime' in context and context['realtime'].get('name'):
                name = context['realtime']['name']
            else:
                # 最后从映射表获取
                name = STOCK_NAME_MAP.get(code, f'股票{code}')
        
        # 如果模型不可用，返回默认结果
        if not self.is_available():
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary='AI 分析功能未启用（未配置 API Key）',
                risk_warning='请配置 LLM API Key（GEMINI_API_KEY/ANTHROPIC_API_KEY/OPENAI_API_KEY）后重试',
                success=False,
                error_message='LLM API Key 未配置',
                model_used=None,
            )
        
        try:
            config = get_config()
            model_name = config.litellm_model or "unknown"
            logger.info(f"========== AI 分析 {name}({code}) (3-Step) ==========")
            logger.info(f"[LLM配置] 模型: {model_name}")

            request_delay = config.gemini_request_delay

            # Step 1: Technical analysis (price, MA, MACD, RSI, volume)
            if request_delay > 0:
                time.sleep(request_delay)
            step1_raw = self._analyze_step("Step1-技术面", self._format_step1_prompt(context, name))
            step1_result = self._parse_step_json(step1_raw, "Step1")

            # Step 2: Structural validation (chip + backtest)
            if step1_result and request_delay > 0:
                time.sleep(request_delay)
            step2_result = None
            if step1_result:
                step2_raw = self._analyze_step(
                    "Step2-筹码回测", self._format_step2_prompt(context, name, step1_result), max_tokens=2048
                )
                step2_result = self._parse_step_json(step2_raw, "Step2")

            # Step 3: Final decision (news + guardrails + all prior results)
            if step1_result and request_delay > 0:
                time.sleep(request_delay)
            if step1_result:
                step3_raw = self._analyze_step(
                    "Step3-最终决策",
                    self._format_step3_prompt(context, name, news_context, step1_result, step2_result or {}),
                    max_tokens=8192,
                )
                result = self._parse_response(step3_raw, code, name)
                result.step1_technical = step1_result
                result.step2_validation = step2_result
            else:
                # Step 1 failed, fall back to single-call mode
                logger.warning(f"[LLM] Step 1 失败，回退到单步模式")
                prompt = self._format_prompt(context, name, news_context)
                response_text, model_used, llm_usage = self._call_litellm(
                    prompt, {"temperature": config.gemini_temperature, "max_output_tokens": 8192}
                )
                result = self._parse_response(response_text, code, name)
                result.model_used = model_used
                persist_llm_usage(llm_usage, model_used, call_type="analysis", stock_code=code)

            result.raw_response = getattr(result, 'raw_response', '')
            result.search_performed = bool(news_context)
            result.market_snapshot = self._build_market_snapshot(context)

            # Fill data_perspective from context when LLM skips it
            _fill_data_perspective_from_context(result, context)
            if 'chip' not in context:
                mark_chip_structure_missing(result)

            decision_guardrails = self._build_decision_guardrails(context)
            if decision_guardrails:
                self._apply_decision_guardrails(result, decision_guardrails)

            # Content integrity (optional)
            if config.report_integrity_enabled:
                pass_integrity, missing_fields = self._check_content_integrity(result)
                if not pass_integrity:
                    self._apply_placeholder_fill(result, missing_fields)
                    logger.warning(
                        "[LLM完整性] 必填字段缺失 %s，已占位补全",
                        missing_fields,
                    )

            logger.info(f"[LLM解析] {name}({code}) 分析完成: {result.trend_prediction}, 评分 {result.sentiment_score}")

            return result
            
        except Exception as e:
            logger.error(f"AI 分析 {name}({code}) 失败: {e}")
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary=f'分析过程出错: {str(e)[:100]}',
                risk_warning='分析失败，请稍后重试或手动分析',
                success=False,
                error_message=str(e),
                model_used=None,
            )
    
    def _format_prompt(
        self, 
        context: Dict[str, Any], 
        name: str,
        news_context: Optional[str] = None
    ) -> str:
        """
        格式化分析提示词（决策仪表盘 v2.0）
        
        包含：技术指标、实时行情（量比/换手率）、筹码分布、趋势分析、新闻
        
        Args:
            context: 技术面数据上下文（包含增强数据）
            name: 股票名称（默认值，可能被上下文覆盖）
            news_context: 预先搜索的新闻内容
        """
        code = context.get('code', 'Unknown')
        
        # 优先使用上下文中的股票名称（从 realtime_quote 获取）
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'股票{code}':
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')
            
        today = context.get('today', {})
        
        # ========== 构建决策仪表盘格式的输入 ==========
        prompt = f"""# 决策仪表盘分析请求

## 📊 股票基础信息
| 项目 | 数据 |
|------|------|
| 股票代码 | **{code}** |
| 股票名称 | **{stock_name}** |
| 分析日期 | {context.get('date', '未知')} |

---

## 🏷️ 筹码分布（核心参考，权重最高）

"""

        # 筹码分布数据 — 提前到最前面，确保 LLM 优先考虑
        if 'chip' in context:
            chip = context['chip']
            profit_ratio = chip.get('profit_ratio', 0)
            prompt += f"""
| 指标 | 数值 | 健康标准 |
|------|------|----------|
| 数据日期 | {chip.get('date', 'N/A')} | 缓存数据仅作参考 |
| 数据来源 | {chip.get('source', 'N/A')} | cache: 开头表示使用缓存 |
| **获利比例** | **{profit_ratio:.1%}** | <10%极重套牢 / 10-70%正常 / >70%获利回吐风险 |
| 平均成本 | {chip.get('avg_cost', 'N/A')} 元 | 现价应高于5-15% |
| 90%筹码集中度 | {chip.get('concentration_90', 0):.2%} | <8%高度集中 <15%集中 |
| 70%筹码集中度 | {chip.get('concentration_70', 0):.2%} | |
| 筹码状态 | {chip.get('chip_status', '未知')} | |
"""
        else:
            prompt += """
⚠️ 筹码分布数据未获取。请在 `chip_structure` 中输出"数据不足/无法判断"，严禁编造具体数值。
"""
        prompt += f"""
---

## 📈 技术面数据

### 今日行情
| 指标 | 数值 |
|------|------|
| 收盘价 | {today.get('close', 'N/A')} 元 |
| 开盘价 | {today.get('open', 'N/A')} 元 |
| 最高价 | {today.get('high', 'N/A')} 元 |
| 最低价 | {today.get('low', 'N/A')} 元 |
| 涨跌幅 | {today.get('pct_chg', 'N/A')}% |
| 成交量 | {self._format_volume(today.get('volume'))} |
| 成交额 | {self._format_amount(today.get('amount'))} |

### 均线系统（关键判断指标）
| 均线 | 数值 | 说明 |
|------|------|------|
| MA5 | {today.get('ma5', 'N/A')} | 短期趋势线 |
| MA10 | {today.get('ma10', 'N/A')} | 中短期趋势线 |
| MA20 | {today.get('ma20', 'N/A')} | 中期趋势线 |
| 均线形态 | {context.get('ma_status', '未知')} | 多头/空头/缠绕 |
"""
        
        # 添加实时行情数据（量比、换手率等）
        if 'realtime' in context:
            rt = context['realtime']
            prompt += f"""
### 实时行情增强数据
| 指标 | 数值 | 解读 |
|------|------|------|
| 当前价格 | {rt.get('price', 'N/A')} 元 | |
| **量比** | **{rt.get('volume_ratio', 'N/A')}** | {rt.get('volume_ratio_desc', '')} |
| **换手率** | **{rt.get('turnover_rate', 'N/A')}%** | |
| 市盈率(动态) | {rt.get('pe_ratio', 'N/A')} | |
| 市净率 | {rt.get('pb_ratio', 'N/A')} | |
| 总市值 | {self._format_amount(rt.get('total_mv'))} | |
| 流通市值 | {self._format_amount(rt.get('circ_mv'))} | |
| 60日涨跌幅 | {rt.get('change_60d', 'N/A')}% | 中期表现 |
"""
         
        # 添加趋势分析结果（客观数据展示）
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            bias_warning = "🚨 超过5%，严禁追高！" if trend.get('bias_ma5', 0) > 5 else "✅ 安全范围"
            prompt += f"""
### 趋势分析预判（基于交易理念）
| 指标 | 数值 | 判定 |
|------|------|------|
| 趋势状态 | {trend.get('trend_status', '未知')} | |
| 均线排列 | {trend.get('ma_alignment', '未知')} | MA5>MA10>MA20为多头 |
| 趋势强度 | {trend.get('trend_strength', 0)}/100 | |
| **乖离率(MA5)** | **{trend.get('bias_ma5', 0):+.2f}%** | {bias_warning} |
| 乖离率(MA10) | {trend.get('bias_ma10', 0):+.2f}% | |
| 量能状态 | {trend.get('volume_status', '未知')} | {trend.get('volume_trend', '')} |
| 系统信号 | {trend.get('buy_signal', '未知')} | |
| 系统评分 | {trend.get('signal_score', 0)}/100 | |

#### 系统分析理由
**买入理由**：
{chr(10).join('- ' + r for r in trend.get('signal_reasons', ['无'])) if trend.get('signal_reasons') else '- 无'}

**风险因素**：
{chr(10).join('- ' + r for r in trend.get('risk_factors', ['无'])) if trend.get('risk_factors') else '- 无'}

### MACD 与 RSI 指标
| 指标 | 数值 | 解读 |
|------|------|------|
| MACD DIF | {trend.get('macd_dif', 'N/A')} | |
| MACD DEA | {trend.get('macd_dea', 'N/A')} | |
| MACD BAR | {trend.get('macd_bar', 'N/A')} | {trend.get('macd_signal', 'N/A')} |
| MACD 状态 | {trend.get('macd_status', 'N/A')} | |
| RSI(6) | {trend.get('rsi_6', 'N/A')} | |
| RSI(12) | {trend.get('rsi_12', 'N/A')} | |
| RSI(24) | {trend.get('rsi_24', 'N/A')} | {trend.get('rsi_signal', 'N/A')} |
| RSI 状态 | {trend.get('rsi_status', 'N/A')} | |
"""
         
        # 添加昨日对比数据
        if 'yesterday' in context:
            volume_change = context.get('volume_change_ratio', 'N/A')
            prompt += f"""
### 量价变化
- 成交量较昨日变化：{volume_change}倍
- 价格较昨日变化：{context.get('price_change_ratio', 'N/A')}%
"""
        
        # 添加量化回测数据（提前到新闻之前，提升权重）
        config = get_config()
        if getattr(config, "quant_backtest_prompt_enabled", False):
            try:
                from src.services.quant_context_service import (
                    format_quant_summary_for_prompt,
                    get_quant_summary_by_code,
                )

                quant_summary = get_quant_summary_by_code(
                    code,
                    getattr(config, "quant_backtest_summary_path", "reports/stock_pool_backtest_summary.json"),
                )
                quant_prompt = format_quant_summary_for_prompt(quant_summary)
                if quant_prompt:
                    prompt += quant_prompt
            except Exception as e:
                logger.debug("Quant backtest prompt context skipped: %s", e)

        if getattr(config, "backtest_compare_enabled", False):
            try:
                from src.services.quant_context_service import (
                    format_backtest_comparison_for_prompt,
                    get_backtest_comparison_by_code,
                    get_cloud_summary_by_code,
                )

                comparison = get_backtest_comparison_by_code(
                    code,
                    getattr(config, "backtest_comparison_path", "reports/backtest_comparison.json"),
                )
                cloud_summary = get_cloud_summary_by_code(
                    code,
                    getattr(config, "cloud_backtest_summary_path", "reports/cloud_backtest_summary.json"),
                )
                prompt += format_backtest_comparison_for_prompt(comparison, cloud_summary)
            except Exception as e:
                logger.debug("Backtest comparison prompt context skipped: %s", e)

        decision_guardrails_prompt = self._format_decision_guardrails(self._build_decision_guardrails(context))
        if decision_guardrails_prompt:
            prompt += decision_guardrails_prompt
        
        # 添加新闻搜索结果（重点区域）
        prompt += """
---

## 📰 舆情情报
"""
        if news_context:
            if context.get('is_index_etf'):
                prompt += f"""
以下是 **{stock_name}({code})** 近7日的新闻搜索结果，请重点提取：
1. 📢 **最新动态**：指数走势、跟踪表现、净值变化、规模变动
2. 🚨 **风险警报**：跟踪误差扩大、市场波动、流动性风险
3. ✨ **利好催化**：指数成分股走强、宏观利好、资金流入

```
{news_context}
```
"""
            else:
                prompt += f"""
以下是 **{stock_name}({code})** 近7日的新闻搜索结果，请重点提取：
1. 🚨 **风险警报**：减持、处罚、利空
2. 🎯 **利好催化**：业绩、合同、政策
3. 📊 **业绩预期**：年报预告、业绩快报

```
{news_context}
```
"""
        else:
            prompt += """
 未搜索到该股票近期的相关新闻。请主要依据技术面和筹码数据进行分析。
"""

        # 注入缺失数据警告
        if context.get('data_missing'):
            prompt += """
⚠️ **数据缺失警告**
由于接口限制，当前无法获取完整的实时行情和技术指标数据。
请 **忽略上述表格中的 N/A 数据**，重点依据 **【📰 舆情情报】** 中的新闻进行基本面和情绪面分析。
在回答技术面问题（如均线、乖离率）时，请直接说明“数据缺失，无法判断”，**严禁编造数据**。
"""

        # 明确的输出要求
        prompt += f"""
---

## ✅ 分析任务

请为 **{stock_name}({code})** 生成【决策仪表盘】，严格按照 JSON 格式输出。
"""
        if context.get('is_index_etf'):
            prompt += """
> ⚠️ **指数/ETF 分析约束**：该标的为指数跟踪型 ETF 或市场指数。
> - 风险分析仅关注：**指数走势、跟踪误差、市场流动性**
> - 严禁将基金公司的诉讼、声誉、高管变动纳入风险警报
> - 业绩预期基于**指数成分股整体表现**，而非基金公司财报
> - `risk_alerts` 中不得出现基金管理人相关的公司经营风险

"""
        prompt += f"""
### ⚠️ 重要：输出正确的股票名称格式
正确的股票名称格式为“股票名称（股票代码）”，例如“贵州茅台（600519）”。
如果上方显示的股票名称为"股票{code}"或不正确，请在分析开头**明确输出该股票的正确中文全称**。

### 重点关注（必须明确回答）：
1. ❓ 是否满足 MA5>MA10>MA20 多头排列？
2. ❓ 当前乖离率是否在安全范围内（<5%）？—— 超过5%必须标注"严禁追高"
3. ❓ 量能是否配合（缩量回调/放量突破）？
4. ❓ **筹码结构是否健康？**（获利比例 < 10% 深套 / 10-70% 正常 / > 70% 获利回吐风险；集中度 < 15% 为集中；现价与平均成本关系）
5. ❓ **量化回测数据是否支持？**（回测有效且盈利 → 可加分；回测高风险/跑输基准 → 降级；样本不足 → 提示待验证）
6. ❓ 消息面有无重大利空？（减持、处罚、业绩变脸等）

### 决策仪表盘要求：
- **股票名称**：必须输出正确的中文全称（如"贵州茅台"而非"股票600519"）
- **核心结论**：一句话说清该买/该卖/该等
- **持仓分类建议**：空仓者怎么做 vs 持仓者怎么做
- **具体狙击点位**：买入价、止损价、目标价（精确到分）
- **检查清单**：每项用 ✅/⚠️/❌ 标记

 请输出完整的 JSON 格式决策仪表盘。"""
        
        return prompt

    def _format_step1_prompt(self, context: Dict[str, Any], name: str) -> str:
        """Step 1: Pure technical analysis — price, trend, momentum. No chip, no backtest, no news."""
        code = context.get('code', 'Unknown')
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'股票{code}':
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')
        today = context.get('today', {})

        prompt = f"""# 技术面趋势分析（第一步）

## 📊 股票基础信息
| 项目 | 数据 |
|------|------|
| 股票代码 | **{code}** |
| 股票名称 | **{stock_name}** |
| 分析日期 | {context.get('date', '未知')} |

## 📈 K线与均线

### 今日行情
| 指标 | 数值 |
|------|------|
| 收盘价 | {today.get('close', 'N/A')} 元 |
| 开盘价 | {today.get('open', 'N/A')} 元 |
| 最高价 | {today.get('high', 'N/A')} 元 |
| 最低价 | {today.get('low', 'N/A')} 元 |
| 涨跌幅 | {today.get('pct_chg', 'N/A')}% |
| 成交量 | {self._format_volume(today.get('volume'))} |
| 成交额 | {self._format_amount(today.get('amount'))} |

### 均线系统
| 均线 | 数值 | 说明 |
|------|------|------|
| MA5 | {today.get('ma5', 'N/A')} | 短期趋势线 |
| MA10 | {today.get('ma10', 'N/A')} | 中短期趋势线 |
| MA20 | {today.get('ma20', 'N/A')} | 中期趋势线 |
| 均线形态 | {context.get('ma_status', '未知')} | 多头/空头/缠绕 |
"""

        # 实时行情
        if 'realtime' in context:
            rt = context['realtime']
            prompt += f"""
### 实时行情
| 指标 | 数值 | 解读 |
|------|------|------|
| 当前价格 | {rt.get('price', 'N/A')} 元 | |
| **量比** | **{rt.get('volume_ratio', 'N/A')}** | {rt.get('volume_ratio_desc', '')} |
| **换手率** | **{rt.get('turnover_rate', 'N/A')}%** | |
| 市盈率(动态) | {rt.get('pe_ratio', 'N/A')} | |
| 市净率 | {rt.get('pb_ratio', 'N/A')} | |
| 总市值 | {self._format_amount(rt.get('total_mv'))} | |
| 流通市值 | {self._format_amount(rt.get('circ_mv'))} | |
| 60日涨跌幅 | {rt.get('change_60d', 'N/A')}% | 中期表现 |
"""

        # 趋势分析 + MACD/RSI
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            bias_warning = "🚨 超过5%，严禁追高！" if trend.get('bias_ma5', 0) > 5 else "✅ 安全范围"
            prompt += f"""
### 趋势与动量
| 指标 | 数值 | 判定 |
|------|------|------|
| 趋势状态 | {trend.get('trend_status', '未知')} | |
| 均线排列 | {trend.get('ma_alignment', '未知')} | MA5>MA10>MA20为多头 |
| 趋势强度 | {trend.get('trend_strength', 0)}/100 | |
| **乖离率(MA5)** | **{trend.get('bias_ma5', 0):+.2f}%** | {bias_warning} |
| 乖离率(MA10) | {trend.get('bias_ma10', 0):+.2f}% | |
| 量能状态 | {trend.get('volume_status', '未知')} | {trend.get('volume_trend', '')} |

### MACD 与 RSI
| 指标 | 数值 | 解读 |
|------|------|------|
| MACD DIF | {trend.get('macd_dif', 'N/A')} | |
| MACD DEA | {trend.get('macd_dea', 'N/A')} | |
| MACD BAR | {trend.get('macd_bar', 'N/A')} | {trend.get('macd_signal', 'N/A')} |
| MACD 状态 | {trend.get('macd_status', 'N/A')} | |
| RSI(6) | {trend.get('rsi_6', 'N/A')} | |
| RSI(12) | {trend.get('rsi_12', 'N/A')} | |
| RSI(24) | {trend.get('rsi_24', 'N/A')} | {trend.get('rsi_signal', 'N/A')} |
| RSI 状态 | {trend.get('rsi_status', 'N/A')} | |
"""

        # 量价变化
        if 'yesterday' in context:
            prompt += f"""
### 量价变化 vs 昨日
- 成交量较昨日变化：{context.get('volume_change_ratio', 'N/A')}倍
- 价格较昨日变化：{context.get('price_change_ratio', 'N/A')}%
"""

        prompt += f"""
---

## ✅ 第一步任务：纯技术面趋势判断

⚠️ 本步骤**只分析价格、均线、量能、MACD/RSI**——不看筹码、不看回测、不看新闻。

输出 JSON：
```json
{{
    "trend_direction": "上升趋势/下降趋势/横盘震荡",
    "ma_alignment": "多头排列/空头排列/均线缠绕",
    "ma_values": "MA5=X > MA10=Y > MA20=Z 的完整数值",
    "bias_ma5_pct": 0.0,
    "bias_status": "安全(<2%)/警戒(2-5%)/追高风险(>5%)",
    "momentum": {{
        "macd_signal": "多头/空头/金叉/死叉/底背离/顶背离",
        "rsi_value": 50,
        "rsi_status": "正常/超买/超卖"
    }},
    "volume_assessment": "缩量回调/放量突破/缩量下跌/放量下跌/量价背离",
    "volume_ratio": 0.0,
    "turnover_rate": 0.0,
    "trend_strength_100": 0,
    "technical_risk_signals": ["风险信号"],
    "preliminary_bias": "偏多/偏空/中性",
    "key_observations": "2-3个最重要的技术面发现，逗号分隔"
}}
```
"""
        return prompt

    def _format_step2_prompt(
        self, context: Dict[str, Any], name: str,
        step1_result: Dict[str, Any],
    ) -> str:
        """Step 2: Structural validation — chip distribution + quant backtest, referencing step 1."""
        code = context.get('code', 'Unknown')
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'股票{code}':
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')

        step1_json = json.dumps(step1_result, ensure_ascii=False, indent=2)

        prompt = f"""# 结构验证分析（第二步）

## 📊 股票基础信息
| 项目 | 数据 |
|------|------|
| 股票代码 | **{code}** |
| 股票名称 | **{stock_name}** |

## 📋 第一步结果：技术面趋势判断

```json
{step1_json}
```

---

## 🏷️ 筹码分布数据
"""
        if 'chip' in context:
            chip = context['chip']
            profit_ratio = chip.get('profit_ratio', 0)
            prompt += f"""
| 指标 | 数值 | 健康标准 |
|------|------|----------|
| 数据日期 | {chip.get('date', 'N/A')} | 缓存数据仅作参考 |
| 数据来源 | {chip.get('source', 'N/A')} | cache: 开头表示使用缓存 |
| **获利比例** | **{profit_ratio:.1%}** | <10%极重套牢 / 10-70%正常 / >70%获利回吐风险 |
| 平均成本 | {chip.get('avg_cost', 'N/A')} 元 | |
| 90%筹码集中度 | {chip.get('concentration_90', 0):.2%} | <8%高度集中 <15%集中 |
| 70%筹码集中度 | {chip.get('concentration_70', 0):.2%} | |
| 筹码状态 | {chip.get('chip_status', '未知')} | |
"""
        else:
            prompt += "\n⚠️ 筹码分布数据未获取。\n"

        prompt += """
---

## 📈 量化回测数据
"""
        config = get_config()
        if getattr(config, "quant_backtest_prompt_enabled", False):
            try:
                from src.services.quant_context_service import (
                    format_quant_summary_for_prompt,
                    get_quant_summary_by_code,
                )
                quant_summary = get_quant_summary_by_code(
                    code,
                    getattr(config, "quant_backtest_summary_path", "reports/stock_pool_backtest_summary.json"),
                )
                quant_prompt = format_quant_summary_for_prompt(quant_summary)
                if quant_prompt:
                    prompt += quant_prompt
                else:
                    prompt += "\n⚠️ 该标的无可用回测数据。\n"
            except Exception as e:
                logger.debug("Quant backtest skipped: %s", e)
                prompt += "\n⚠️ 回测数据获取失败。\n"
        else:
            prompt += "\n回测功能未启用。\n"

        prompt += f"""
---

## ✅ 第二步任务：验证第一趋势判断

你的任务是：
1. **筹码验证**：获利比例/集中度/成本关系是否支持第一步的趋势判断？如果趋势偏多但获利>70%，必须标注"筹码质疑看多信号"
2. **回测验证**：历史回测结果是否支持当前判断？夏普/最大回撤/策略收益是确认还是推翻？
3. **综合调整**：在第一 `preliminary_bias` 基础上，根据筹码和回测调整方向。例如筹码深套+回测高风险 → 即使趋势偏多也应转为中性或偏空

输出 JSON：
```json
{{
    "chip_validation": "确认看多/质疑看多/确认看空/质疑看空/中性",
    "chip_reasoning": "筹码验证的详细理由（100字以内）",
    "chip_health": "健康/一般/危险",
    "chip_profit_level": "极重套牢/深套/正常/获利较高/极度获利",
    "backtest_validation": "确认/质疑/数据不足",
    "backtest_reasoning": "回测验证的详细理由（100字以内）",
    "adjusted_bias": "偏多/偏空/中性",
    "adjusted_sentiment": 0,
    "risk_signals_from_validation": ["结构风险信号"],
    "key_validation_points": "2-3个最重要的验证发现"
}}
```

⚠️ 如果回测样本不足，`backtest_validation` 应为 "数据不足"，只在 `backtest_reasoning` 中说明，不得将其解读为策略亏损。
"""
        return prompt

    def _format_step3_prompt(
        self, context: Dict[str, Any], name: str,
        news_context: Optional[str],
        step1_result: Dict[str, Any],
        step2_result: Dict[str, Any],
    ) -> str:
        """Step 3: Final synthesis — news + guardrails + step1/2 results → decision dashboard."""
        code = context.get('code', 'Unknown')
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'股票{code}':
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')

        step1_json = json.dumps(step1_result, ensure_ascii=False, indent=2)
        step2_json = json.dumps(step2_result, ensure_ascii=False, indent=2)

        prompt = f"""# 最终决策仪表盘（第三步）

## 📊 股票基础信息
| 项目 | 数据 |
|------|------|
| 股票代码 | **{code}** |
| 股票名称 | **{stock_name}** |

## 📋 前两步分析结果

### Step 1：技术面趋势
```json
{step1_json}
```

### Step 2：结构验证（筹码 + 回测）
```json
{step2_json}
```

---

"""

        # Decision guardrails
        decision_guardrails = self._format_decision_guardrails(self._build_decision_guardrails(context))
        if decision_guardrails:
            prompt += f"{decision_guardrails}\n---\n\n"

        # News
        prompt += "## 📰 舆情情报\n"
        if news_context:
            if context.get('is_index_etf'):
                prompt += f"""
以下是 **{stock_name}({code})** 近7日新闻，重点提取指数走势、跟踪误差、流动性风险：

```
{news_context}
```
"""
            else:
                prompt += f"""
以下是 **{stock_name}({code})** 近7日新闻，重点提取减持、处罚、业绩预告等风险信号：

```
{news_context}
```
"""
        else:
            prompt += "\n未搜索到该股票近期新闻。\n"

        prompt += f"""
---

## ✅ 第三步任务：输出完整决策仪表盘 JSON

综合 Step1 技术面趋势、Step2 筹码+回测验证、舆情情报，为 **{stock_name}({code})** 生成完整决策仪表盘。

⚠️ **硬约束**：
- `sentiment_score` = Step2 的 `adjusted_sentiment` 基础上，根据新闻加减分（减持/利空 -10~20，利好 +5~15）
- `chip_structure.profit_ratio` 必须使用 Step2 的 `chip_profit_level`、`chip_health`
- `operation_advice` 必须与 Step2 的 `adjusted_bias` 一致，不得矛盾
- 回测高风险/样本不足时，不得给出"买入"建议

{"" if not context.get('is_index_etf') else "> ⚠️ 指数/ETF 约束：严禁将基金公司诉讼、高管变动纳入 risk_alerts"}

请输出完整的 JSON 格式决策仪表盘。"""
        return prompt

    def _analyze_step(self, label: str, prompt: str, max_tokens: int = 2048) -> Optional[str]:
        """Run a single step LLM call, return raw response text."""
        config = get_config()
        generation_config = {
            "temperature": config.gemini_temperature,
            "max_output_tokens": max_tokens,
        }
        logger.info(f"[{label}] Prompt 长度: {len(prompt)} 字符")
        response_text, model_used, usage = self._call_litellm(prompt, generation_config)
        logger.info(f"[{label}完成] 模型: {model_used}, tokens: {usage.get('total_tokens', 'N/A')}")
        return response_text

    def _parse_step_json(self, response_text: str, step_label: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from a step response. Returns None on failure."""
        try:
            cleaned = response_text
            if '```json' in cleaned:
                cleaned = cleaned.replace('```json', '').replace('```', '')
            elif '```' in cleaned:
                cleaned = cleaned.replace('```', '')
            json_start = cleaned.find('{')
            json_end = cleaned.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(cleaned[json_start:json_end])
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[{step_label}] JSON 解析失败 ({e}), raw: {response_text[:300]}")
        return None

    def _build_decision_guardrails(self, context: Dict[str, Any]) -> Dict[str, Any] | None:
        config = get_config()
        if not getattr(config, "decision_rule_enabled", False):
            return None

        code = context.get('code', 'Unknown')
        try:
            from src.services.decision_rule_service import evaluate_decision_rules
            from src.services.quant_context_service import get_backtest_comparison_by_code, get_quant_summary_by_code

            quant_summary = get_quant_summary_by_code(
                code,
                getattr(config, "quant_backtest_summary_path", "reports/stock_pool_backtest_summary.json"),
            )
            backtest_comparison = None
            if getattr(config, "backtest_compare_enabled", False):
                backtest_comparison = get_backtest_comparison_by_code(
                    code,
                    getattr(config, "backtest_comparison_path", "reports/backtest_comparison.json"),
                )
            return evaluate_decision_rules(
                code=code,
                quant_summary=quant_summary,
                backtest_comparison=backtest_comparison,
                technical_context=context,
                config_path=getattr(config, "decision_rule_config_path", "config/decision_rules.yaml"),
            )
        except Exception as e:
            logger.debug("Decision rule guardrails skipped: %s", e)
            return None

    def _format_decision_guardrails(self, guardrails: Dict[str, Any] | None) -> str:
        if not guardrails:
            return ""
        try:
            from src.services.decision_rule_service import format_decision_guardrails_for_prompt

            return format_decision_guardrails_for_prompt(guardrails)
        except Exception as e:
            logger.debug("Decision rule prompt context skipped: %s", e)
            return ""

    def _apply_decision_guardrails(self, result: AnalysisResult, guardrails: Dict[str, Any] | None) -> None:
        if not guardrails:
            return
        try:
            from src.services.decision_rule_service import apply_decision_guardrails_to_result

            if apply_decision_guardrails_to_result(result, guardrails):
                logger.info("Decision guardrails downgraded %s to %s", result.code, result.operation_advice)
        except Exception as e:
            logger.debug("Decision rule result guardrails skipped: %s", e)
    
    def _format_volume(self, volume: Optional[float]) -> str:
        """格式化成交量显示"""
        if volume is None:
            return 'N/A'
        if volume >= 1e8:
            return f"{volume / 1e8:.2f} 亿股"
        elif volume >= 1e4:
            return f"{volume / 1e4:.2f} 万股"
        else:
            return f"{volume:.0f} 股"
    
    def _format_amount(self, amount: Optional[float]) -> str:
        """格式化成交额显示"""
        if amount is None:
            return 'N/A'
        if amount >= 1e8:
            return f"{amount / 1e8:.2f} 亿元"
        elif amount >= 1e4:
            return f"{amount / 1e4:.2f} 万元"
        else:
            return f"{amount:.0f} 元"

    def _format_percent(self, value: Optional[float]) -> str:
        """格式化百分比显示"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return 'N/A'

    def _format_price(self, value: Optional[float]) -> str:
        """格式化价格显示"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return 'N/A'

    def _build_market_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """构建当日行情快照（展示用）"""
        today = context.get('today', {}) or {}
        realtime = context.get('realtime', {}) or {}
        yesterday = context.get('yesterday', {}) or {}

        prev_close = yesterday.get('close')
        close = today.get('close')
        high = today.get('high')
        low = today.get('low')

        amplitude = None
        change_amount = None
        if prev_close not in (None, 0) and high is not None and low is not None:
            try:
                amplitude = (float(high) - float(low)) / float(prev_close) * 100
            except (TypeError, ValueError, ZeroDivisionError):
                amplitude = None
        if prev_close is not None and close is not None:
            try:
                change_amount = float(close) - float(prev_close)
            except (TypeError, ValueError):
                change_amount = None

        snapshot = {
            "date": context.get('date', '未知'),
            "close": self._format_price(close),
            "open": self._format_price(today.get('open')),
            "high": self._format_price(high),
            "low": self._format_price(low),
            "prev_close": self._format_price(prev_close),
            "pct_chg": self._format_percent(today.get('pct_chg')),
            "change_amount": self._format_price(change_amount),
            "amplitude": self._format_percent(amplitude),
            "volume": self._format_volume(today.get('volume')),
            "amount": self._format_amount(today.get('amount')),
        }

        if realtime:
            snapshot.update({
                "price": self._format_price(realtime.get('price')),
                "volume_ratio": realtime.get('volume_ratio', 'N/A'),
                "turnover_rate": self._format_percent(realtime.get('turnover_rate')),
                "source": getattr(realtime.get('source'), 'value', realtime.get('source', 'N/A')),
            })

        return snapshot

    def _check_content_integrity(self, result: AnalysisResult) -> Tuple[bool, List[str]]:
        """Delegate to module-level check_content_integrity."""
        return check_content_integrity(result)

    def _build_integrity_complement_prompt(self, missing_fields: List[str]) -> str:
        """Build complement instruction for missing mandatory fields."""
        lines = ["### 补全要求：请在上方分析基础上补充以下必填内容，并输出完整 JSON："]
        for f in missing_fields:
            if f == "sentiment_score":
                lines.append("- sentiment_score: 0-100 综合评分")
            elif f == "operation_advice":
                lines.append("- operation_advice: 买入/加仓/持有/减仓/卖出/观望")
            elif f == "analysis_summary":
                lines.append("- analysis_summary: 综合分析摘要")
            elif f == "dashboard.core_conclusion.one_sentence":
                lines.append("- dashboard.core_conclusion.one_sentence: 一句话决策")
            elif f == "dashboard.intelligence.risk_alerts":
                lines.append("- dashboard.intelligence.risk_alerts: 风险警报列表（可为空数组）")
            elif f == "dashboard.battle_plan.sniper_points.stop_loss":
                lines.append("- dashboard.battle_plan.sniper_points.stop_loss: 止损价")
        return "\n".join(lines)

    def _build_integrity_retry_prompt(
        self,
        base_prompt: str,
        previous_response: str,
        missing_fields: List[str],
    ) -> str:
        """Build retry prompt using the previous response as the complement baseline."""
        complement = self._build_integrity_complement_prompt(missing_fields)
        previous_output = previous_response.strip()
        return "\n\n".join([
            base_prompt,
            "### 上一次输出如下，请在该输出基础上补齐缺失字段，并重新输出完整 JSON。不要省略已有字段：",
            previous_output,
            complement,
        ])

    def _apply_placeholder_fill(self, result: AnalysisResult, missing_fields: List[str]) -> None:
        """Delegate to module-level apply_placeholder_fill."""
        apply_placeholder_fill(result, missing_fields)

    def _parse_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """
        解析 Gemini 响应（决策仪表盘版）
        
        尝试从响应中提取 JSON 格式的分析结果，包含 dashboard 字段
        如果解析失败，尝试智能提取或返回默认结果
        """
        try:
            # 清理响应文本：移除 markdown 代码块标记
            cleaned_text = response_text
            if '```json' in cleaned_text:
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
            elif '```' in cleaned_text:
                cleaned_text = cleaned_text.replace('```', '')
            
            # 尝试找到 JSON 内容
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                
                # 尝试修复常见的 JSON 问题
                json_str = self._fix_json_string(json_str)
                
                data = json.loads(json_str)

                # Schema validation (lenient: on failure, continue with raw dict)
                try:
                    AnalysisReportSchema.model_validate(data)
                except Exception as e:
                    logger.warning(
                        "LLM report schema validation failed, continuing with raw dict: %s",
                        str(e)[:100],
                    )

                # 提取 dashboard 数据
                dashboard = data.get('dashboard', None)

                # 优先使用 AI 返回的股票名称（如果原名称无效或包含代码）
                ai_stock_name = data.get('stock_name')
                if ai_stock_name and (name.startswith('股票') or name == code or 'Unknown' in name):
                    name = ai_stock_name

                # 解析所有字段，使用默认值防止缺失
                # 解析 decision_type，如果没有则根据 operation_advice 推断
                decision_type = data.get('decision_type', '')
                if not decision_type:
                    op = data.get('operation_advice', '持有')
                    if op in ['买入', '加仓', '强烈买入']:
                        decision_type = 'buy'
                    elif op in ['卖出', '减仓', '强烈卖出']:
                        decision_type = 'sell'
                    else:
                        decision_type = 'hold'
                
                return AnalysisResult(
                    code=code,
                    name=name,
                    # 核心指标
                    sentiment_score=int(data.get('sentiment_score', 50)),
                    trend_prediction=data.get('trend_prediction', '震荡'),
                    operation_advice=data.get('operation_advice', '持有'),
                    decision_type=decision_type,
                    confidence_level=data.get('confidence_level', '中'),
                    # 决策仪表盘
                    dashboard=dashboard,
                    # 走势分析
                    trend_analysis=data.get('trend_analysis', ''),
                    short_term_outlook=data.get('short_term_outlook', ''),
                    medium_term_outlook=data.get('medium_term_outlook', ''),
                    # 技术面
                    technical_analysis=data.get('technical_analysis', ''),
                    ma_analysis=data.get('ma_analysis', ''),
                    volume_analysis=data.get('volume_analysis', ''),
                    pattern_analysis=data.get('pattern_analysis', ''),
                    # 基本面
                    fundamental_analysis=data.get('fundamental_analysis', ''),
                    sector_position=data.get('sector_position', ''),
                    company_highlights=data.get('company_highlights', ''),
                    # 情绪面/消息面
                    news_summary=data.get('news_summary', ''),
                    market_sentiment=data.get('market_sentiment', ''),
                    hot_topics=data.get('hot_topics', ''),
                    # 综合
                    analysis_summary=data.get('analysis_summary', '分析完成'),
                    key_points=data.get('key_points', ''),
                    risk_warning=data.get('risk_warning', ''),
                    buy_reason=data.get('buy_reason', ''),
                    # 元数据
                    search_performed=data.get('search_performed', False),
                    data_sources=data.get('data_sources', '技术面数据'),
                    success=True,
                )
            else:
                # 没有找到 JSON，尝试从纯文本中提取信息
                logger.warning(f"无法从响应中提取 JSON，使用原始文本分析")
                return self._parse_text_response(response_text, code, name)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}，尝试从文本提取")
            return self._parse_text_response(response_text, code, name)
    
    def _fix_json_string(self, json_str: str) -> str:
        """修复常见的 JSON 格式问题"""
        import re
        
        # 移除注释
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # 修复尾随逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 确保布尔值是小写
        json_str = json_str.replace('True', 'true').replace('False', 'false')
        
        # fix by json-repair
        json_str = repair_json(json_str)
        
        return json_str
    
    def _parse_text_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """从纯文本响应中尽可能提取分析信息"""
        # 尝试识别关键词来判断情绪
        sentiment_score = 50
        trend = '震荡'
        advice = '持有'
        
        text_lower = response_text.lower()
        
        # 简单的情绪识别
        positive_keywords = ['看多', '买入', '上涨', '突破', '强势', '利好', '加仓', 'bullish', 'buy']
        negative_keywords = ['看空', '卖出', '下跌', '跌破', '弱势', '利空', '减仓', 'bearish', 'sell']
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if positive_count > negative_count + 1:
            sentiment_score = 65
            trend = '看多'
            advice = '买入'
            decision_type = 'buy'
        elif negative_count > positive_count + 1:
            sentiment_score = 35
            trend = '看空'
            advice = '卖出'
            decision_type = 'sell'
        else:
            decision_type = 'hold'
        
        # 截取前500字符作为摘要
        summary = response_text[:500] if response_text else '无分析结果'
        
        return AnalysisResult(
            code=code,
            name=name,
            sentiment_score=sentiment_score,
            trend_prediction=trend,
            operation_advice=advice,
            decision_type=decision_type,
            confidence_level='低',
            analysis_summary=summary,
            key_points='JSON解析失败，仅供参考',
            risk_warning='分析结果可能不准确，建议结合其他信息判断',
            raw_response=response_text,
            success=True,
        )
    
    def batch_analyze(
        self, 
        contexts: List[Dict[str, Any]],
        delay_between: float = 2.0
    ) -> List[AnalysisResult]:
        """
        批量分析多只股票
        
        注意：为避免 API 速率限制，每次分析之间会有延迟
        
        Args:
            contexts: 上下文数据列表
            delay_between: 每次分析之间的延迟（秒）
            
        Returns:
            AnalysisResult 列表
        """
        results = []
        
        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"等待 {delay_between} 秒后继续...")
                time.sleep(delay_between)
            
            result = self.analyze(context)
            results.append(result)
        
        return results


# 便捷函数
def get_analyzer() -> GeminiAnalyzer:
    """获取 LLM 分析器实例"""
    return GeminiAnalyzer()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    # 模拟上下文数据
    test_context = {
        'code': '600519',
        'date': '2026-01-09',
        'today': {
            'open': 1800.0,
            'high': 1850.0,
            'low': 1780.0,
            'close': 1820.0,
            'volume': 10000000,
            'amount': 18200000000,
            'pct_chg': 1.5,
            'ma5': 1810.0,
            'ma10': 1800.0,
            'ma20': 1790.0,
            'volume_ratio': 1.2,
        },
        'ma_status': '多头排列 📈',
        'volume_change_ratio': 1.3,
        'price_change_ratio': 1.5,
    }
    
    analyzer = GeminiAnalyzer()
    
    if analyzer.is_available():
        print("=== AI 分析测试 ===")
        result = analyzer.analyze(test_context)
        print(f"分析结果: {result.to_dict()}")
    else:
        print("Gemini API 未配置，跳过测试")
