# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 主调度程序
===================================

职责：
1. 协调各模块完成股票分析流程
2. 实现低并发的线程池调度
3. 全局异常处理，确保单股失败不影响整体
4. 提供命令行入口

使用方式：
    python main.py              # 正常运行
    python main.py --debug      # 调试模式
    python main.py --dry-run    # 仅获取数据不分析

交易理念（已融入分析）：
- 严进策略：不追高，乖离率 > 5% 不买入
- 趋势交易：只做 MA5>MA10>MA20 多头排列
- 效率优先：关注筹码集中度好的股票
- 买点偏好：缩量回踩 MA5/MA10 支撑
"""
import os
import warnings
from src.config import setup_env

# Suppress known-harmless shutdown noise (Python 3.12 __del__ + logging teardown)
warnings.filterwarnings("ignore", message=".*utcfromtimestamp.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*There is no current event loop.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*unclosed event loop.*", category=ResourceWarning)
setup_env()

# 代理配置 - 通过 USE_PROXY 环境变量控制，默认关闭
# GitHub Actions 环境自动跳过代理配置
if os.getenv("GITHUB_ACTIONS") != "true" and os.getenv("USE_PROXY", "false").lower() == "true":
    # 本地开发环境，启用代理（可在 .env 中配置 PROXY_HOST 和 PROXY_PORT）
    proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
    proxy_port = os.getenv("PROXY_PORT", "10809")
    proxy_url = f"http://{proxy_host}:{proxy_port}"
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url

import argparse
import logging
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from data_provider.base import canonical_stock_code
from src.core.pipeline import StockAnalysisPipeline
from src.core.market_review import run_market_review
from src.webui_frontend import prepare_webui_frontend_assets
from src.config import get_config, Config
from src.logging_config import setup_logging


logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='A股自选股智能分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python main.py                    # 正常运行
  python main.py --debug            # 调试模式
  python main.py --dry-run          # 仅获取数据，不进行 AI 分析
  python main.py --stocks 600519,000001  # 指定分析特定股票
  python main.py --no-notify        # 不发送推送通知
  python main.py --single-notify    # 启用单股推送模式（每分析完一只立即推送）
  python main.py --schedule         # 启用定时任务模式
  python main.py --market-review    # 仅运行大盘复盘
        '''
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式，输出详细日志'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅获取数据，不进行 AI 分析'
    )

    parser.add_argument(
        '--stocks',
        type=str,
        help='指定要分析的股票代码，逗号分隔（覆盖配置文件）'
    )

    parser.add_argument(
        '--no-notify',
        action='store_true',
        help='不发送推送通知'
    )

    parser.add_argument(
        '--single-notify',
        action='store_true',
        help='启用单股推送模式：每分析完一只股票立即推送，而不是汇总推送'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='并发线程数（默认使用配置值）'
    )

    parser.add_argument(
        '--schedule',
        action='store_true',
        help='启用定时任务模式，每日定时执行'
    )

    parser.add_argument(
        '--no-run-immediately',
        action='store_true',
        help='定时任务启动时不立即执行一次'
    )

    parser.add_argument(
        '--market-review',
        action='store_true',
        help='仅运行大盘复盘分析'
    )

    parser.add_argument(
        '--no-market-review',
        action='store_true',
        help='跳过大盘复盘分析'
    )

    parser.add_argument(
        '--force-run',
        action='store_true',
        help='跳过交易日检查，强制执行全量分析（Issue #373）'
    )

    parser.add_argument(
        '--webui',
        action='store_true',
        help='启动 Web 管理界面'
    )

    parser.add_argument(
        '--webui-only',
        action='store_true',
        help='仅启动 Web 服务，不执行自动分析'
    )

    parser.add_argument(
        '--serve',
        action='store_true',
        help='启动 FastAPI 后端服务（同时执行分析任务）'
    )

    parser.add_argument(
        '--serve-only',
        action='store_true',
        help='仅启动 FastAPI 后端服务，不自动执行分析'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='FastAPI 服务端口（默认 8000）'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='FastAPI 服务监听地址（默认 0.0.0.0）'
    )

    parser.add_argument(
        '--no-context-snapshot',
        action='store_true',
        help='不保存分析上下文快照'
    )

    # === Backtest ===
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='运行回测（对历史分析结果进行评估）'
    )

    parser.add_argument(
        '--backtest-code',
        type=str,
        default=None,
        help='仅回测指定股票代码'
    )

    parser.add_argument(
        '--backtest-days',
        type=int,
        default=None,
        help='回测评估窗口（交易日数，默认使用配置）'
    )

    parser.add_argument(
        '--backtest-force',
        action='store_true',
        help='强制回测（即使已有回测结果也重新计算）'
    )

    # === Personal trades ===
    parser.add_argument(
        '--add-trade',
        action='store_true',
        help='Record a personal trade (use with --code --direction --price --volume --date '
             '[--name] [--trigger] [--followed-rules/--no-followed-rules] [--notes])',
    )
    parser.add_argument(
        '--code',
        type=str,
        default=None,
        help='Stock code (for --add-trade, --list-trades, --trade-stats)',
    )
    parser.add_argument(
        '--direction',
        type=str,
        choices=['buy', 'sell'],
        default=None,
        help='Trade direction: buy or sell',
    )
    parser.add_argument(
        '--price',
        type=float,
        default=None,
        help='Trade price',
    )
    parser.add_argument(
        '--volume',
        type=int,
        default=None,
        help='Trade volume (shares)',
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Trade date (YYYY-MM-DD, default today)',
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Stock name (optional)',
    )
    parser.add_argument(
        '--trigger',
        type=str,
        default=None,
        help='Exit trigger: stop_loss / take_profit / manual / signal',
    )
    parser.add_argument(
        '--no-followed-rules',
        action='store_true',
        help='Mark this trade as not following the rules',
    )
    parser.add_argument(
        '--notes',
        type=str,
        default=None,
        help='Trade notes',
    )
    parser.add_argument(
        '--list-trades',
        action='store_true',
        help='List personal trade records',
    )
    parser.add_argument(
        '--delete-trade',
        type=int,
        default=None,
        help='Delete a trade record by ID',
    )
    parser.add_argument(
        '--trade-stats',
        action='store_true',
        help='Show per-stock trade stats (paired buy/sell P&L)',
    )

    return parser.parse_args()


def _compute_trading_day_filter(
    config: Config,
    args: argparse.Namespace,
    stock_codes: List[str],
) -> Tuple[List[str], Optional[str], bool]:
    """
    Compute filtered stock list and effective market review region (Issue #373).

    Returns:
        (filtered_codes, effective_region, should_skip_all)
        - effective_region None = use config default (check disabled)
        - effective_region '' = all relevant markets closed, skip market review
        - should_skip_all: skip entire run when no stocks and no market review to run
    """
    force_run = getattr(args, 'force_run', False)
    if force_run or not getattr(config, 'trading_day_check_enabled', True):
        return (stock_codes, None, False)

    from src.core.trading_calendar import (
        get_market_for_stock,
        get_open_markets_today,
        compute_effective_region,
    )

    open_markets = get_open_markets_today()
    filtered_codes = []
    for code in stock_codes:
        mkt = get_market_for_stock(code)
        if mkt in open_markets or mkt is None:
            filtered_codes.append(code)

    if config.market_review_enabled and not getattr(args, 'no_market_review', False):
        effective_region = compute_effective_region(
            getattr(config, 'market_review_region', 'cn') or 'cn', open_markets
        )
    else:
        effective_region = None

    should_skip_all = (not filtered_codes) and (effective_region or '') == ''
    return (filtered_codes, effective_region, should_skip_all)


def run_full_analysis(
    config: Config,
    args: argparse.Namespace,
    stock_codes: Optional[List[str]] = None
):
    """
    执行完整的分析流程（个股 + 大盘复盘）

    这是定时任务调用的主函数
    """
    try:
        # Issue #529: Hot-reload STOCK_LIST from .env on each scheduled run
        if stock_codes is None:
            config.refresh_stock_list()

        # Issue #373: Trading day filter (per-stock, per-market)
        effective_codes = stock_codes if stock_codes is not None else config.stock_list
        filtered_codes, effective_region, should_skip = _compute_trading_day_filter(
            config, args, effective_codes
        )
        if should_skip:
            logger.info(
                "今日所有相关市场均为非交易日，跳过执行。可使用 --force-run 强制执行。"
            )
            return
        if set(filtered_codes) != set(effective_codes):
            skipped = set(effective_codes) - set(filtered_codes)
            logger.info("今日休市股票已跳过: %s", skipped)
        stock_codes = filtered_codes

        # 命令行参数 --single-notify 覆盖配置（#55）
        if getattr(args, 'single_notify', False):
            config.single_stock_notify = True

        # Issue #190: 个股与大盘复盘合并推送
        merge_notification = (
            getattr(config, 'merge_email_notification', False)
            and config.market_review_enabled
            and not getattr(args, 'no_market_review', False)
            and not config.single_stock_notify
        )

        # 创建调度器
        save_context_snapshot = None
        if getattr(args, 'no_context_snapshot', False):
            save_context_snapshot = False
        query_id = uuid.uuid4().hex
        pipeline = StockAnalysisPipeline(
            config=config,
            max_workers=args.workers,
            query_id=query_id,
            query_source="cli",
            save_context_snapshot=save_context_snapshot
        )

        # 1. 运行个股分析
        results = pipeline.run(
            stock_codes=stock_codes,
            dry_run=args.dry_run,
            send_notification=not args.no_notify,
            merge_notification=merge_notification
        )

        decision_report_content = ""
        if results:
            try:
                from src.services.decision_service import write_decision_report

                decision_report = write_decision_report(
                    results,
                    output_path=getattr(config, 'decision_report_output_path', 'reports/daily_decision_report.md'),
                    report_type=getattr(config, 'decision_report_type', 'daily_action_list'),
                )
                logger.info(
                    "每日行动清单已生成: %s",
                    decision_report.get('files', {}).get('markdown'),
                )
                decision_report_path = decision_report.get('files', {}).get('markdown')
                if decision_report_path:
                    decision_report_content = Path(decision_report_path).read_text(encoding='utf-8')
                if (
                    decision_report_content
                    and not args.no_notify
                    and not merge_notification
                    and pipeline.notifier.is_available()
                ):
                    if pipeline.notifier.send(decision_report_content, email_send_to_all=True):
                        logger.info("每日行动清单已推送")
                    else:
                        logger.warning("每日行动清单推送失败")
            except Exception as e:
                logger.warning("每日行动清单生成失败: %s", e)

        # Issue #128: 分析间隔 - 在个股分析和大盘分析之间添加延迟
        analysis_delay = getattr(config, 'analysis_delay', 0)
        if (
            analysis_delay > 0
            and config.market_review_enabled
            and not args.no_market_review
            and effective_region != ''
        ):
            logger.info(f"等待 {analysis_delay} 秒后执行大盘复盘（避免API限流）...")
            time.sleep(analysis_delay)

        # 2. 运行大盘复盘（如果启用且不是仅个股模式）
        market_report = ""
        if (
            config.market_review_enabled
            and not args.no_market_review
            and effective_region != ''
        ):
            review_result = run_market_review(
                notifier=pipeline.notifier,
                analyzer=pipeline.analyzer,
                search_service=pipeline.search_service,
                send_notification=not args.no_notify,
                merge_notification=merge_notification,
                override_region=effective_region,
            )
            # 如果有结果，赋值给 market_report 用于后续飞书文档生成
            if review_result:
                market_report = review_result

        # Issue #190: 合并推送（个股+大盘复盘）
        if merge_notification and (results or market_report) and not args.no_notify:
            parts = []
            if market_report:
                parts.append(f"# 📈 大盘复盘\n\n{market_report}")
            if results:
                dashboard_content = pipeline.notifier.generate_aggregate_report(
                    results,
                    getattr(config, 'report_type', 'simple'),
                )
                parts.append(f"# 🚀 个股决策仪表盘\n\n{dashboard_content}")
            if decision_report_content:
                parts.append(f"# ✅ 每日行动清单\n\n{decision_report_content}")
            if parts:
                combined_content = "\n\n---\n\n".join(parts)
                if pipeline.notifier.is_available():
                    if pipeline.notifier.send(combined_content, email_send_to_all=True):
                        logger.info("已合并推送（个股+大盘复盘）")
                    else:
                        logger.warning("合并推送失败")

        # 输出摘要
        if results:
            logger.info("\n===== 分析结果摘要 =====")
            for r in sorted(results, key=lambda x: x.sentiment_score, reverse=True):
                emoji = r.get_emoji()
                logger.info(
                    f"{emoji} {r.name}({r.code}): {r.operation_advice} | "
                    f"评分 {r.sentiment_score} | {r.trend_prediction}"
                )

        logger.info("\n任务执行完成")

        # === 新增：生成飞书云文档 ===
        try:
            from src.feishu_doc import FeishuDocManager

            feishu_doc = FeishuDocManager()
            if feishu_doc.is_configured() and (results or market_report):
                logger.info("正在创建飞书云文档...")

                # 1. 准备标题 "01-01 13:01大盘复盘"
                tz_cn = timezone(timedelta(hours=8))
                now = datetime.now(tz_cn)
                doc_title = f"{now.strftime('%Y-%m-%d %H:%M')} 大盘复盘"

                # 2. 准备内容 (拼接个股分析和大盘复盘)
                full_content = ""

                # 添加大盘复盘内容（如果有）
                if market_report:
                    full_content += f"# 📈 大盘复盘\n\n{market_report}\n\n---\n\n"

                # 添加个股决策仪表盘（使用 NotificationService 生成，按 report_type 分支）
                if results:
                    dashboard_content = pipeline.notifier.generate_aggregate_report(
                        results,
                        getattr(config, 'report_type', 'simple'),
                    )
                    full_content += f"# 🚀 个股决策仪表盘\n\n{dashboard_content}"

                if decision_report_content:
                    if full_content:
                        full_content += "\n\n---\n\n"
                    full_content += f"# ✅ 每日行动清单\n\n{decision_report_content}"

                # 3. 创建文档
                doc_url = feishu_doc.create_daily_doc(doc_title, full_content)
                if doc_url:
                    logger.info(f"飞书云文档创建成功: {doc_url}")
                    # 可选：将文档链接也推送到群里
                    if not args.no_notify:
                        pipeline.notifier.send(f"[{now.strftime('%Y-%m-%d %H:%M')}] 复盘文档创建成功: {doc_url}")

        except Exception as e:
            logger.error(f"飞书文档生成失败: {e}")

        # === Auto backtest ===
        try:
            if getattr(config, 'backtest_enabled', False):
                from src.services.backtest_service import BacktestService

                logger.info("开始自动回测...")
                service = BacktestService()
                stats = service.run_backtest(
                    force=False,
                    eval_window_days=getattr(config, 'backtest_eval_window_days', 10),
                    min_age_days=getattr(config, 'backtest_min_age_days', 14),
                    limit=200,
                )
                logger.info(
                    f"自动回测完成: processed={stats.get('processed')} saved={stats.get('saved')} "
                    f"completed={stats.get('completed')} insufficient={stats.get('insufficient')} errors={stats.get('errors')}"
                )
        except Exception as e:
            logger.warning(f"自动回测失败（已忽略）: {e}")

    except Exception as e:
        logger.exception(f"分析流程执行失败: {e}")


def start_api_server(host: str, port: int, config: Config) -> None:
    """
    在后台线程启动 FastAPI 服务
    
    Args:
        host: 监听地址
        port: 监听端口
        config: 配置对象
    """
    import threading
    import uvicorn

    def run_server():
        level_name = (config.log_level or "INFO").lower()
        uvicorn.run(
            "api.app:app",
            host=host,
            port=port,
            log_level=level_name,
            log_config=None,
        )

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logger.info(f"FastAPI 服务已启动: http://{host}:{port}")


def _is_truthy_env(var_name: str, default: str = "true") -> bool:
    """Parse common truthy / falsy environment values."""
    value = os.getenv(var_name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}

def start_bot_stream_clients(config: Config) -> None:
    """Start bot stream clients when enabled in config."""
    # 启动钉钉 Stream 客户端
    if config.dingtalk_stream_enabled:
        try:
            from bot.platforms import start_dingtalk_stream_background, DINGTALK_STREAM_AVAILABLE
            if DINGTALK_STREAM_AVAILABLE:
                if start_dingtalk_stream_background():
                    logger.info("[Main] Dingtalk Stream client started in background.")
                else:
                    logger.warning("[Main] Dingtalk Stream client failed to start.")
            else:
                logger.warning("[Main] Dingtalk Stream enabled but SDK is missing.")
                logger.warning("[Main] Run: pip install dingtalk-stream")
        except Exception as exc:
            logger.error(f"[Main] Failed to start Dingtalk Stream client: {exc}")

    # 启动飞书 Stream 客户端
    if getattr(config, 'feishu_stream_enabled', False):
        try:
            from bot.platforms import start_feishu_stream_background, FEISHU_SDK_AVAILABLE
            if FEISHU_SDK_AVAILABLE:
                if start_feishu_stream_background():
                    logger.info("[Main] Feishu Stream client started in background.")
                else:
                    logger.warning("[Main] Feishu Stream client failed to start.")
            else:
                logger.warning("[Main] Feishu Stream enabled but SDK is missing.")
                logger.warning("[Main] Run: pip install lark-oapi")
        except Exception as exc:
            logger.error(f"[Main] Failed to start Feishu Stream client: {exc}")


def _ensure_search_proxy_running(config: Config) -> None:
    """Auto-start SearXNG on localhost if configured and not already running."""
    priority = getattr(config, "news_search_source_priority", "") or ""
    if "searxng" not in priority.lower():
        return

    searxng_urls = getattr(config, "searxng_base_urls", []) or []
    local_url = None
    for url in searxng_urls:
        if "127.0.0.1" in url or "localhost" in url:
            local_url = url.rstrip("/")
            break
    if not local_url:
        return

    start_script = str(Path(__file__).parent / "scripts" / "start_searxng.sh")
    if not Path(start_script).exists():
        logger.warning(f"SearXNG 启动脚本不存在: {start_script}")
        return

    import subprocess

    try:
        subprocess.run(["bash", start_script], timeout=20, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        logger.warning("SearXNG 启动脚本超时")
    except Exception as exc:
        logger.error(f"SearXNG 启动失败: {exc}")
        return

    for _ in range(30):
        time.sleep(1)
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"{local_url}/search?format=json&q=test", timeout=3)
            if resp.status == 200:
                logger.info(f"SearXNG 已就绪: {local_url}")
                return
        except Exception:
            pass

    logger.warning("SearXNG 启动超时，搜索将回退到其他数据源")


def _ensure_searxng_running(config: Config) -> None:
    """Auto-start search proxy on localhost if configured and not already running."""
    priority = getattr(config, "news_search_source_priority", "") or ""
    if "searxng" not in priority.lower():
        return
    _ensure_search_proxy_running(config)


def _cleanup_litellm() -> None:
    """Close all litellm/httpx pools before interpreter teardown to suppress __del__ noise."""
    import logging
    try:
        try:
            import litellm
            for attr in ("_async_http_handler", "_sync_http_handler", "client"):
                handler = getattr(litellm, attr, None)
                if handler and hasattr(handler, "close"):
                    try:
                        handler.close()
                    except Exception:
                        pass
        except Exception:
            pass
        # Also close the litellm client's transport pools
        try:
            import httpx
            for obj in ("_async_client", "_sync_client"):
                client = getattr(litellm, obj, None) if "litellm" in dir() else None
                if client and hasattr(client, "_transport"):
                    try:
                        client._transport.close()
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception:
        pass
    # Kill search proxy started by this run
    try:
        import subprocess
        subprocess.run(["pkill", "-f", "search_proxy.py"], timeout=5, capture_output=True)
    except Exception:
        pass

    # Suppress __del__ noise: raise root level so httpx/httpcore debug logs
    # are silently dropped even after cleanup. Do NOT call logging.shutdown()
    # because that invalidates handlers and causes AttributeError in __del__.
    logging.root.setLevel(logging.CRITICAL + 1)


def handle_personal_trades(args: argparse.Namespace) -> int:
    """Handle personal trade journal CLI commands."""
    from src.storage import get_db, PersonalTrade
    from datetime import date as date_type

    db = get_db()

    # --add-trade
    if args.add_trade:
        if not args.code or not args.direction or args.price is None or args.volume is None:
            logger.error("--add-trade requires --code, --direction, --price, --volume")
            return 1
        trade_date = date_type.today()
        if args.date:
            trade_date = date_type.fromisoformat(args.date)
        followed = not args.no_followed_rules
        tid = db.save_personal_trade(
            code=args.code.upper(),
            name=args.name,
            direction=args.direction,
            price=args.price,
            volume=args.volume,
            trade_date=trade_date,
            trigger=args.trigger,
            followed_rules=followed,
            notes=args.notes,
        )
        logger.info(f"Trade recorded: id={tid} {args.code} {args.direction} {args.price} x{args.volume}")
        return 0

    # --list-trades
    if args.list_trades:
        trades = db.get_personal_trades(code=args.code)
        if not trades:
            logger.info("No trade records found.")
            return 0
        print(f"\n{'ID':<6} {'Date':<12} {'Code':<8} {'Name':<8} {'Dir':<4} {'Price':>8} {'Vol':>6} {'Trigger':<12} {'Rules':<6} {'Notes'}")
        print("-" * 100)
        for t in trades:
            td = t  # now returns dict
            rules_mark = "yes" if td.get('followed_rules') else "NO"
            print(
                f"{td.get('id',''):<6} {td.get('trade_date',''):<12} {td['code']:<8} {(td.get('name') or ''):<8} "
                f"{td['direction']:<4} {td['price']:>8.3f} {td['volume']:>6} "
                f"{(td.get('trigger') or ''):<12} {rules_mark:<6} {td.get('notes') or ''}"
            )
        return 0

    # --delete-trade
    if args.delete_trade is not None:
        ok = db.delete_personal_trade(args.delete_trade)
        if ok:
            logger.info(f"Trade {args.delete_trade} deleted.")
        else:
            logger.warning(f"Trade {args.delete_trade} not found.")
        return 0

    # --trade-stats
    if args.trade_stats:
        pairs = db.get_personal_trade_stats(code=args.code)
        if not pairs:
            logger.info("No completed trade pairs found.")
            return 0
        print(f"\n{'Code':<8} {'Name':<10} {'Buy Date':<12} {'Buy':>8} {'Sell Date':<12} {'Sell':>8} {'Vol':>6} {'P&L%':>8} {'Trigger':<12} {'Rules'}")
        print("-" * 110)
        for p in pairs:
            rules_mark = "yes" if p['followed_rules'] else "NO"
            print(
                f"{p['code']:<8} {p.get('name',''):<10} {p['buy_date']:<12} {p['buy_price']:>8.3f} "
                f"{p['sell_date']:<12} {p['sell_price']:>8.3f} {p['volume']:>6} "
                f"{p['pnl_pct']:>+7.2f}% {(p.get('trigger') or ''):<12} {rules_mark}"
            )
        print()
        return 0

    return -1  # no trade command matched (proceed to normal flow)


def main() -> int:
    """
    主入口函数

    Returns:
        退出码（0 表示成功）
    """
    # Suppress harmless httpx/__del__ noise during Python shutdown by closing
    # litellm's connection pool before the logging system is torn down.
    import atexit
    atexit.register(_cleanup_litellm)

    # 解析命令行参数
    args = parse_arguments()

    # 加载配置（在设置日志前加载，以获取日志目录）
    config = get_config()

    # 配置日志（输出到控制台和文件）
    setup_logging(
        log_prefix="stock_analysis",
        debug=args.debug,
        log_dir=config.log_dir,
        console_info_modules={"__main__", "src.notification", "src.core.pipeline"},
    )

    logger.info("=" * 60)
    logger.info("A股自选股智能分析系统 启动")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # === 个人交易日记（独立于分析流程） ===
    trade_result = handle_personal_trades(args)
    if trade_result >= 0:
        return trade_result

    # 验证配置
    warnings = config.validate()
    for warning in warnings:
        logger.warning(warning)

    # Auto-start search proxy if configured and not already running
    _ensure_search_proxy_running(config)

    # 解析股票列表（统一为大写 Issue #355）
    stock_codes = None
    if args.stocks:
        stock_codes = [canonical_stock_code(c) for c in args.stocks.split(',') if (c or "").strip()]
        logger.info(f"使用命令行指定的股票列表: {stock_codes}")

    # === 处理 --webui / --webui-only 参数，映射到 --serve / --serve-only ===
    if args.webui:
        args.serve = True
    if args.webui_only:
        args.serve_only = True

    # 兼容旧版 WEBUI_ENABLED 环境变量
    if config.webui_enabled and not (args.serve or args.serve_only):
        args.serve = True

    # === 启动 Web 服务 (如果启用) ===
    start_serve = (args.serve or args.serve_only) and os.getenv("GITHUB_ACTIONS") != "true"

    # 兼容旧版 WEBUI_HOST/WEBUI_PORT：如果用户未通过 --host/--port 指定，则使用旧变量
    if start_serve:
        if args.host == '0.0.0.0' and os.getenv('WEBUI_HOST'):
            args.host = os.getenv('WEBUI_HOST')
        if args.port == 8000 and os.getenv('WEBUI_PORT'):
            args.port = int(os.getenv('WEBUI_PORT'))

    bot_clients_started = False
    if start_serve:
        if not prepare_webui_frontend_assets():
            logger.warning("前端静态资源未就绪，继续启动 FastAPI 服务（Web 页面可能不可用）")
        try:
            start_api_server(host=args.host, port=args.port, config=config)
            bot_clients_started = True
        except Exception as e:
            logger.error(f"启动 FastAPI 服务失败: {e}")

    if bot_clients_started:
        start_bot_stream_clients(config)

    # === 仅 Web 服务模式：不自动执行分析 ===
    if args.serve_only:
        logger.info("模式: 仅 Web 服务")
        logger.info(f"Web 服务运行中: http://{args.host}:{args.port}")
        logger.info("通过 /api/v1/analysis/stock/{code} 接口触发分析")
        logger.info(f"API 文档: http://{args.host}:{args.port}/docs")
        logger.info("按 Ctrl+C 退出...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n用户中断，程序退出")
        return 0

    try:
        # 模式0: 回测
        if getattr(args, 'backtest', False):
            logger.info("模式: 回测")
            from src.services.backtest_service import BacktestService

            service = BacktestService()
            stats = service.run_backtest(
                code=getattr(args, 'backtest_code', None),
                force=getattr(args, 'backtest_force', False),
                eval_window_days=getattr(args, 'backtest_days', None),
            )
            logger.info(
                f"回测完成: processed={stats.get('processed')} saved={stats.get('saved')} "
                f"completed={stats.get('completed')} insufficient={stats.get('insufficient')} errors={stats.get('errors')}"
            )
            return 0

        # 模式1: 仅大盘复盘
        if args.market_review:
            from src.analyzer import GeminiAnalyzer
            from src.core.market_review import run_market_review
            from src.notification import NotificationService
            from src.search_service import SearchService

            # Issue #373: Trading day check for market-review-only mode.
            # Do NOT use _compute_trading_day_filter here: that helper checks
            # config.market_review_enabled, which would wrongly block an
            # explicit --market-review invocation when the flag is disabled.
            effective_region = None
            if not getattr(args, 'force_run', False) and getattr(config, 'trading_day_check_enabled', True):
                from src.core.trading_calendar import get_open_markets_today, compute_effective_region as _compute_region
                open_markets = get_open_markets_today()
                effective_region = _compute_region(
                    getattr(config, 'market_review_region', 'cn') or 'cn', open_markets
                )
                if effective_region == '':
                    logger.info("今日大盘复盘相关市场均为非交易日，跳过执行。可使用 --force-run 强制执行。")
                    return 0

            logger.info("模式: 仅大盘复盘")
            notifier = NotificationService()

            # 初始化搜索服务和分析器（如果有配置）
            search_service = None
            analyzer = None

            if config.bocha_api_keys or config.tavily_api_keys or config.brave_api_keys or config.serpapi_keys or config.minimax_api_keys or config.searxng_base_urls:
                search_service = SearchService(
                    bocha_keys=config.bocha_api_keys,
                    tavily_keys=config.tavily_api_keys,
                    brave_keys=config.brave_api_keys,
                    serpapi_keys=config.serpapi_keys,
                    minimax_keys=config.minimax_api_keys,
                    searxng_base_urls=config.searxng_base_urls,
                    news_max_age_days=config.news_max_age_days,
                    source_priority=getattr(
                        config,
                        'news_search_source_priority',
                        'bocha,tavily,brave,serpapi,minimax,searxng',
                    ),
                )

            if config.gemini_api_key or config.openai_api_key:
                analyzer = GeminiAnalyzer(api_key=config.gemini_api_key)
                if not analyzer.is_available():
                    logger.warning("AI 分析器初始化后不可用，请检查 API Key 配置")
                    analyzer = None
            else:
                logger.warning("未检测到 API Key (Gemini/OpenAI)，将仅使用模板生成报告")

            run_market_review(
                notifier=notifier,
                analyzer=analyzer,
                search_service=search_service,
                send_notification=not args.no_notify,
                override_region=effective_region,
            )
            return 0

        # 模式2: 定时任务模式
        if args.schedule or config.schedule_enabled:
            logger.info("模式: 定时任务")
            logger.info(f"每日执行时间: {config.schedule_time}")

            # Determine whether to run immediately:
            # Command line arg --no-run-immediately overrides config if present.
            # Otherwise use config (defaults to True).
            should_run_immediately = config.schedule_run_immediately
            if getattr(args, 'no_run_immediately', False):
                should_run_immediately = False

            logger.info(f"启动时立即执行: {should_run_immediately}")

            from src.scheduler import run_with_schedule

            def scheduled_task():
                run_full_analysis(config, args, stock_codes)

            run_with_schedule(
                task=scheduled_task,
                schedule_time=config.schedule_time,
                run_immediately=should_run_immediately
            )
            return 0

        # 模式3: 正常单次运行
        if config.run_immediately:
            run_full_analysis(config, args, stock_codes)
        else:
            logger.info("配置为不立即运行分析 (RUN_IMMEDIATELY=false)")

        logger.info("\n程序执行完成")

        return 0

    except KeyboardInterrupt:
        logger.info("\n用户中断，程序退出")
        return 130

    except Exception as e:
        logger.exception(f"程序执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
