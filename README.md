<div align="center">

# 📈 股票智能分析系统

[![GitHub stars](https://img.shields.io/github/stars/ZhuLinsen/daily_stock_analysis?style=social)](https://github.com/ZhuLinsen/daily_stock_analysis/stargazers)
[![CI](https://github.com/ZhuLinsen/daily_stock_analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/ZhuLinsen/daily_stock_analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Ready-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/)

> 🤖 基于 AI 大模型的 A股/港股/美股自选股智能分析系统，每日自动分析并推送「决策仪表盘」到企业微信/飞书/Telegram/邮箱

[**功能特性**](#-功能特性) · [**快速开始**](#-快速开始) · [**推送效果**](#-推送效果) · [**完整指南**](docs/full-guide.md) · [**常见问题**](docs/FAQ.md) · [**更新日志**](docs/CHANGELOG.md)

简体中文 | [English](docs/README_EN.md) | [繁體中文](docs/README_CHT.md)

</div>

## 💖 赞助商 (Sponsors)
<div align="center">
  <a href="https://serpapi.com/baidu-search-api?utm_source=github_daily_stock_analysis" target="_blank">
    <img src="./sources/serpapi_banner_zh.png" alt="轻松抓取搜索引擎上的实时金融新闻数据 - SerpApi" height="160">
  </a>
</div>
<br>


## ✨ 功能特性

| 模块 | 功能 | 说明 |
|------|------|------|
| AI | 决策仪表盘 | 一句话核心结论 + 精确买卖点位 + 操作检查清单 |
| 分析 | 多维度分析 | 技术面（盘中实时 MA/多头排列）+ 筹码分布（支持最近成功缓存兜底）+ 舆情情报 + 实时行情 |
| 市场 | 全球市场 | 支持 A股、港股、美股及美股指数（SPX、DJI、IXIC 等） |
| 策略 | 市场策略系统 | 内置 A股「三段式复盘策略」与美股「Regime Strategy」，输出进攻/均衡/防守或 risk-on/neutral/risk-off 计划，并附“仅供参考，不构成投资建议”提示 |
| 复盘 | 大盘复盘 | 每日市场概览、板块涨跌；支持 cn(A股)/us(美股)/both(两者) 切换 |
| 行动清单 | 股票池执行计划 | 股票池分析完成后生成“今日可执行 / 等待确认 / 禁止追高或剔除”清单，包含空仓/持仓动作、买点、止损、目标价和仓位建议 |
| 智能导入 | 多源导入 | 支持图片、CSV/Excel 文件、剪贴板粘贴；Vision LLM 提取代码+名称；置信度分层确认；名称→代码解析（本地+拼音+AkShare） |
| 回测 | AI 回测验证 | 自动评估历史分析准确率，方向胜率、止盈止损命中率 |
| **Agent 问股** | **策略对话** | **多轮策略问答，支持均线金叉/缠论/波浪等 11 种内置策略，Web/Bot/API 全链路** |
| 推送 | 多渠道通知 | 企业微信、飞书、Telegram、钉钉、邮件、Pushover |
| 自动化 | 定时运行 | GitHub Actions 定时执行，无需服务器 |
| 搜索容错 | 多搜索源 + 配额熔断 | 9 个搜索源（Tavily/SerpAPI/Bocha/Brave/MiniMax/SearXNG/Bing/Google CSE/DuckDuckGo），单 provider 配额耗尽自动熔断、维度内 fallback 到下一个，避免重复浪费 |

> 历史报告详情会优先展示 AI 返回的原始「狙击点位」文本，避免区间价、条件说明等复杂内容在历史回看时被压缩成单个数字。

### 技术栈与数据来源

| 类型 | 支持 |
|------|------|
| AI 模型 | [AIHubMix](https://aihubmix.com/?aff=CfMq)、Gemini、OpenAI 兼容、DeepSeek、通义千问、Claude 等（统一通过 [LiteLLM](https://github.com/BerriAI/litellm) 调用，支持多 Key 负载均衡）|
| 行情数据 | AkShare、Tushare、Pytdx、Baostock、YFinance |
| 筹码分布 | 今日缓存优先；远程优先 Tushare Pro `cyq_perf`（需权限），AkShare 东方财富 `stock_cyq_em` 作为免费兜底 |
| 新闻搜索 | Tavily、SerpAPI、Bocha、Brave、MiniMax、SearXNG、Bing、Google CSE、DuckDuckGo |

> 注：美股历史数据与实时行情统一使用 YFinance，确保复权一致性

### AkShare + vectorbt 本地回测

本仓库提供最小可运行的 A 股本地回测链路，适合在 conda 虚拟环境中快速验证 AKShare 数据获取与 vectorbt 策略回测：

```bash
python src/run_analysis.py
python src/run_stock_pool_backtest.py
python src/daily_report.py
python src/parameter_search.py
```

相关文件：

| 文件 | 说明 |
|------|------|
| `src/data_fetcher.py` | 使用 AKShare 获取 A 股日线并缓存为 Parquet |
| `src/backtest.py` | 使用 vectorbt 执行 MA 均线交叉回测 |
| `src/daily_report.py` | 生成单股票 Markdown 报告和 HTML 图表 |
| `src/parameter_search.py` | 扫描 MA 参数并输出 CSV |
| `src/run_analysis.py` | 一条命令完成数据获取、日报生成和参数扫描 |
| `src/run_stock_pool_backtest.py` | 读取 `stock_pool.txt` 批量运行回测并输出股票池汇总 CSV/JSON |
| `src/export_cloud_backtest.py` | 导出聚宽 JoinQuant 手动回测脚本和云端结果回填模板 |
| `src/compare_backtest_results.py` | 对比本地 vectorbt 和云端手动回填结果，输出一致性报告 |
| `src/render_decision_report.py` | 从分析结果 JSON 渲染每日行动清单 Markdown/JSON |
| `src/services/strategy_backtest_service.py` | 生成供报告、WebUI、通知和 AI 解读消费的标准回测摘要 JSON |
| `stock_pool.txt` | 示例股票池，后续可扩展为批量分析入口 |

数据获取支持普通 A 股代码与带交易所后缀的代码，例如：

```text
000001
600519
159115.SZ
```

股票池批量回测默认使用命令行传入的统一区间；若个别标的上市较晚或需要特殊区间，可在 `stock_pool.txt` 中使用可选格式：

```text
000001
600519
159115.SZ,20250601,20260605
```

其中 ETF 会优先使用 AKShare ETF 历史行情接口，东方财富失败时使用新浪 ETF 历史行情兜底。

可选择将量化回测摘要注入 LLM/DeepSeek prompt，使其影响核心结论、操作点位和持仓建议：

```bash
QUANT_BACKTEST_PROMPT_ENABLED=true python src/inspect_quant_prompt.py
```

配置项：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `QUANT_BACKTEST_ENABLED` | `false` | 是否启用本地量化回测模块入口（默认不改变主流程） |
| `QUANT_BACKTEST_AUTO_RUN` | `false` | 是否在摘要缺失或过期时自动运行股票池回测 |
| `QUANT_BACKTEST_PROMPT_ENABLED` | `false` | 是否把 `stock_pool_backtest_summary.json` 中对应股票的回测摘要注入 LLM prompt |
| `QUANT_BACKTEST_SUMMARY_PATH` | `reports/stock_pool_backtest_summary.json` | 回测摘要 JSON 路径 |
| `QUANT_BACKTEST_USE_STOCK_LIST` | `true` | 是否使用 `.env` 的 `STOCK_LIST` 作为本地/云端量化股票池 |
| `QUANT_BACKTEST_STOCK_POOL_PATH` | `stock_pool.txt` | 股票池文件路径 |
| `QUANT_BACKTEST_STALE_HOURS` | `24` | 摘要超过多少小时视为过期，供自动运行逻辑使用 |
| `QUANT_BACKTEST_LOOKBACK_MONTHS` | `0` | 大于 0 时，本地/云端回测自动用“当前日期往前 N 个月”作为区间 |
| `QUANT_BACKTEST_START_DATE` | `20200101` | 自动运行本地量化回测时使用的开始日期 |
| `QUANT_BACKTEST_END_DATE` | `20240501` | 自动运行本地量化回测时使用的结束日期 |
| `QUANT_BACKTEST_FAST_WINDOW` | `5` | 自动运行本地量化回测时使用的快线 MA 窗口 |
| `QUANT_BACKTEST_SLOW_WINDOW` | `20` | 自动运行本地量化回测时使用的慢线 MA 窗口 |
| `QUANT_BACKTEST_DATA_SOURCE` | `akshare` | 本地 vectorbt 回测历史行情源，可选 `akshare` 或 `joinquant` |
| `JOINQUANT_USERNAME` | - | `QUANT_BACKTEST_DATA_SOURCE=joinquant` 时用于 `jqdatasdk` 登录 |
| `JOINQUANT_PASSWORD` | - | `QUANT_BACKTEST_DATA_SOURCE=joinquant` 时用于 `jqdatasdk` 登录 |
| `CLOUD_BACKTEST_ENABLED` | `false` | 是否启用云端回测结果加载入口 |
| `CLOUD_BACKTEST_PROVIDER` | `joinquant` | 云端回测平台标识 |
| `CLOUD_BACKTEST_SUMMARY_PATH` | `reports/cloud_backtest_summary.json` | 云端回测结果 JSON 路径 |
| `BACKTEST_COMPARE_ENABLED` | `false` | 是否启用本地/云端回测一致性检查入口 |
| `BACKTEST_COMPARISON_PATH` | `reports/backtest_comparison.json` | 本地/云端对比结果 JSON 路径 |
| `BACKTEST_COMPARE_MAX_RETURN_DIFF_PCT` | `5` | 本地/云端策略收益或基准收益最大允许差异 |
| `BACKTEST_COMPARE_MAX_DRAWDOWN_DIFF_PCT` | `5` | 本地/云端最大回撤最大允许差异 |
| `BACKTEST_COMPARE_MAX_TRADE_COUNT_DIFF` | `3` | 本地/云端交易次数最大允许差异 |
| `DECISION_RULE_ENABLED` | `false` | 是否启用决策风控规则入口 |
| `DECISION_RULE_CONFIG_PATH` | `config/decision_rules.yaml` | 决策风控规则配置路径 |
| `DECISION_REPORT_ENABLED` | `false` | 是否启用每日行动清单入口 |
| `DECISION_REPORT_TYPE` | `daily_action_list` | 决策报告类型 |
| `DECISION_REPORT_OUTPUT_PATH` | `reports/daily_decision_report.md` | 决策报告输出路径 |

开启后，prompt 会要求模型在最终建议中考虑收益、基准收益、最大回撤、夏普、交易次数、风险等级和样本不足等约束。
报告会区分两类不可用状态：摘要未命中本次股票时提示重新生成本次股票池摘要；摘要已命中但样本不足或取数失败时提示检查 `QUANT_BACKTEST_DATA_SOURCE`、日期区间和股票代码格式。样本不足不会被当作策略亏损或策略失效。
`QUANT_BACKTEST_ENABLED=true` 时，主流程会在个股分析前检查本地量化摘要；若摘要缺失或过期且 `QUANT_BACKTEST_AUTO_RUN=true`，会自动运行股票池回测生成摘要。云端回测、对比检查、决策风控和行动清单开关当前作为后续模块入口，默认关闭，不影响原有 `.env` 一键运行体验。
如果希望使用聚宽行情做本地 vectorbt 回测，可安装 `jqdatasdk` 并设置 `QUANT_BACKTEST_DATA_SOURCE=joinquant`、`JOINQUANT_USERNAME`、`JOINQUANT_PASSWORD`；该模式只替换本地回测行情源，不自动调用聚宽云端回测任务。
开启 `DECISION_RULE_ENABLED=true` 后，系统会基于 `config/decision_rules.yaml` 对样本不足、跑输基准、高回撤、乖离率过高和数据缺失等场景生成硬约束，并在 LLM 输出越界时自动降级为更保守建议。

可导出聚宽 JoinQuant 手动回测脚本，用于独立校验本地 vectorbt 结果。导出命令只读取本地 `stock_pool.txt` 和配置参数，不访问外部行情数据源：

```bash
python src/export_cloud_backtest.py --platform joinquant --strategy ma_cross
```

输出文件位于 `exports/joinquant/`，同时生成 `reports/cloud_backtest_summary.json` 回填模板。默认使用 `.env` 的 `STOCK_LIST` 作为云端导出股票池；如需单只股票自定义回测区间，可设 `QUANT_BACKTEST_USE_STOCK_LIST=false` 并维护 `stock_pool.txt`。用户可把脚本复制到聚宽免费环境运行，再把总收益、基准收益、最大回撤和交易次数填回模板，供后续本地/云端一致性检查使用。

当本地摘要和云端回填结果都准备好后，可生成一致性对比报告：

```bash
python src/compare_backtest_results.py
```

默认输出 `reports/backtest_comparison.json`。云端结果缺失或未回填不会阻塞程序，只会在对应股票的 `risk_flags` 中标记 `cloud_result_missing`、`cloud_total_return_pct_missing` 等风险标签；收益、回撤和交易次数超过阈值时标记为 `inconsistent`。
开启 `BACKTEST_COMPARE_ENABLED=true` 后，个股 Prompt 会展示本地 vectorbt、聚宽云端结果和一致性判断；若同时开启 `DECISION_RULE_ENABLED=true`，本地/云端不一致或双源都跑输基准时会把最高允许决策限制为观望。云端结果缺失只提示“校验不可用”，不会自动降级。
开启 `DECISION_REPORT_ENABLED=true` 后，主流程会在个股分析完成后写出并推送每日行动清单，把结果分为“今日可执行 / 等待确认 / 禁止追高或剔除”，默认输出 `reports/daily_decision_report.md` 和同名 JSON。清单包含空仓/持仓动作、入场条件、理想/次优买点、止损、目标价、仓位建议和失效条件；若启用个股+大盘合并推送，行动清单会合并进同一条通知和飞书文档。

`python src/run_analysis.py` 默认输出：

```text
reports/{symbol}_{end_date}_ma_report.md
reports/{symbol}_{end_date}_ma_backtest.html
reports/{symbol}_{end_date}_ma_param_search.csv
reports/{symbol}_{end_date}_backtest_summary.json
```

批量股票池回测输出：

```text
reports/stock_pool_backtest_summary.csv
reports/stock_pool_backtest_summary.json
reports/cloud_backtest_summary.json
reports/backtest_comparison.json
reports/daily_decision_report.md
reports/daily_decision_report.json
```

可用以下命令把股票池回测摘要渲染为报告预览：

```bash
python src/render_quant_report.py
```

输出：

```text
reports/quant_backtest_report_preview.md
```

### 内置交易纪律

| 规则 | 说明 |
|------|------|
| 严禁追高 | 乖离率超阈值（默认 5%，可配置）自动提示风险；强势趋势股自动放宽 |
| 趋势交易 | MA5 > MA10 > MA20 多头排列 |
| 精确点位 | 买入价、止损价、目标价 |
| 检查清单 | 每项条件以「满足 / 注意 / 不满足」标记 |
| 新闻时效 | 可配置新闻最大时效（默认 3 天），避免使用过时信息 |

## 🚀 快速开始

### 方式一：GitHub Actions（推荐）

> 5 分钟完成部署，零成本，无需服务器。


#### 1. Fork 本仓库

点击右上角 `Fork` 按钮（顺便点个 Star⭐ 支持一下）

#### 2. 配置 Secrets

`Settings` → `Secrets and variables` → `Actions` → `New repository secret`

**AI 模型配置（至少配置一个）**

> 详细配置说明见 [LLM 配置指南](docs/LLM_CONFIG_GUIDE.md)（三层配置、渠道模式、YAML高级配置、Vision、Agent、排错），GitHub Actions用户也可以实现YAML高级配置。进阶用户可配置 `LITELLM_MODEL`、`LITELLM_FALLBACK_MODELS` 或 `LLM_CHANNELS` 多渠道模式。

> 💡 **推荐 [AIHubMix](https://aihubmix.com/?aff=CfMq)**：一个 Key 即可使用 Gemini、GPT、Claude、DeepSeek 等全球主流模型，无需科学上网，含免费模型（glm-5、gpt-4o-free 等），付费模型高稳定性无限并发。本项目可享 **10% 充值优惠**。

| Secret 名称 | 说明 | 必填 |
|------------|------|:----:|
| `AIHUBMIX_KEY` | [AIHubMix](https://aihubmix.com/?aff=CfMq) API Key，一 Key 切换使用全系模型，免费模型可用 | 可选 |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/) 获取免费 Key（需科学上网） | 可选 |
| `ANTHROPIC_API_KEY` | [Anthropic Claude](https://console.anthropic.com/) API Key | 可选 |
| `ANTHROPIC_MODEL` | Claude 模型（如 `claude-3-5-sonnet-20241022`） | 可选 |
| `OPENAI_API_KEY` | OpenAI 兼容 API Key（支持 DeepSeek、通义千问等） | 可选 |
| `OPENAI_BASE_URL` | OpenAI 兼容 API 地址（如 `https://api.deepseek.com/v1`） | 可选 |
| `OPENAI_MODEL` | 模型名称（如 `gemini-3.1-pro-preview`、`gemini-3-flash-preview`、`gpt-5.2`） | 可选 |
| `OPENAI_VISION_MODEL` | 图片识别专用模型（部分第三方模型不支持图像；不填则用 `OPENAI_MODEL`） | 可选 |

> 注：AI 优先级 Gemini > Anthropic > OpenAI（含 AIHubmix），至少配置一个。`AIHUBMIX_KEY` 无需配置 `OPENAI_BASE_URL`，系统自动适配。图片识别需 Vision 能力模型。DeepSeek 思考模式（deepseek-reasoner、deepseek-r1、qwq、deepseek-chat）按模型名自动识别，无需额外配置。

<details>
<summary><b>通知渠道配置</b>（点击展开，至少配置一个）</summary>


| Secret 名称 | 说明 | 必填 |
|------------|------|:----:|
| `WECHAT_WEBHOOK_URL` | 企业微信 Webhook URL | 可选 |
| `FEISHU_WEBHOOK_URL` | 飞书 Webhook URL | 可选 |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot Token（@BotFather 获取） | 可选 |
| `TELEGRAM_CHAT_ID` | Telegram Chat ID | 可选 |
| `TELEGRAM_MESSAGE_THREAD_ID` | Telegram Topic ID (用于发送到子话题) | 可选 |
| `EMAIL_SENDER` | 发件人邮箱（如 `xxx@qq.com`） | 可选 |
| `EMAIL_PASSWORD` | 邮箱授权码（非登录密码） | 可选 |
| `EMAIL_RECEIVERS` | 收件人邮箱（多个用逗号分隔，留空则发给自己） | 可选 |
| `EMAIL_SENDER_NAME` | 邮件发件人显示名称（默认：daily_stock_analysis股票分析助手） | 可选 |
| `STOCK_GROUP_N` / `EMAIL_GROUP_N` | 股票分组发往不同邮箱（如 `STOCK_GROUP_1=600519,300750` `EMAIL_GROUP_1=user1@example.com`） | 可选 |
| `PUSHPLUS_TOKEN` | PushPlus Token（[获取地址](https://www.pushplus.plus)，国内推送服务） | 可选 |
| `PUSHPLUS_TOPIC` | PushPlus 群组编码（一对多推送，配置后消息推送给群组所有订阅用户） | 可选 |
| `SERVERCHAN3_SENDKEY` | Server酱³ Sendkey（[获取地址](https://sc3.ft07.com/)，手机APP推送服务） | 可选 |
| `CUSTOM_WEBHOOK_URLS` | 自定义 Webhook（支持钉钉等，多个用逗号分隔） | 可选 |
| `CUSTOM_WEBHOOK_BEARER_TOKEN` | 自定义 Webhook 的 Bearer Token（用于需要认证的 Webhook） | 可选 |
| `WEBHOOK_VERIFY_SSL` | Webhook HTTPS 证书校验（默认 true）。设为 false 可支持自签名证书。警告：关闭有严重安全风险，仅限可信内网 | 可选 |
| `SINGLE_STOCK_NOTIFY` | 单股推送模式：设为 `true` 则每分析完一只股票立即推送 | 可选 |
| `REPORT_TYPE` | 报告类型：`simple`(精简)、`full`(完整)、`brief`(3-5句概括)，Docker环境推荐设为 `full` | 可选 |
| `REPORT_SUMMARY_ONLY` | 仅分析结果摘要：设为 `true` 时只推送汇总，不含个股详情 | 可选 |
| `REPORT_TEMPLATES_DIR` | Jinja2 模板目录（相对项目根，默认 `templates`） | 可选 |
| `REPORT_RENDERER_ENABLED` | 启用 Jinja2 模板渲染（默认 `false`，保证零回归） | 可选 |
| `REPORT_INTEGRITY_ENABLED` | 启用报告完整性校验，缺失必填字段时重试或占位补全（默认 `true`） | 可选 |
| `REPORT_INTEGRITY_RETRY` | 完整性校验重试次数（默认 `1`，`0` 表示仅占位不重试） | 可选 |
| `REPORT_HISTORY_COMPARE_N` | 历史信号对比条数，`0` 关闭（默认），`>0` 启用 | 可选 |
| `ANALYSIS_DELAY` | 个股分析和大盘分析之间的延迟（秒），避免API限流，如 `10` | 可选 |
| `MERGE_EMAIL_NOTIFICATION` | 个股与大盘复盘合并推送（默认 false），减少邮件数量 | 可选 |
| `MARKDOWN_TO_IMAGE_CHANNELS` | 将 Markdown 转为图片发送的渠道（逗号分隔）：`telegram,wechat,custom,email` | 可选 |
| `MARKDOWN_TO_IMAGE_MAX_CHARS` | 超过此长度不转图片，避免超大图片（默认 `15000`） | 可选 |
| `MD2IMG_ENGINE` | 转图引擎：`wkhtmltoimage`（默认）或 `markdown-to-file`（emoji 更好） | 可选 |

> 至少配置一个渠道，配置多个则同时推送。图片发送与引擎安装细节请参考 [完整指南](docs/full-guide.md)

</details>

**其他配置**

| Secret 名称 | 说明 | 必填 |
|------------|------|:----:|
| `STOCK_LIST` | 自选股代码，如 `600519,hk00700,AAPL,TSLA` | ✅ |
| `TAVILY_API_KEYS` | [Tavily](https://tavily.com/) 搜索 API（新闻搜索） | 推荐 |
| `MINIMAX_API_KEYS` | [MiniMax](https://platform.minimaxi.com/) Coding Plan Web Search（结构化搜索结果） | 可选 |
| `SERPAPI_API_KEYS` | [SerpAPI](https://serpapi.com/baidu-search-api?utm_source=github_daily_stock_analysis) 全渠道搜索 | 可选 |
| `BOCHA_API_KEYS` | [博查搜索](https://open.bocha.cn/) Web Search API（中文搜索优化，支持AI摘要，多个key用逗号分隔） | 可选 |
| `BRAVE_API_KEYS` | [Brave Search](https://brave.com/search/api/) API（隐私优先，美股优化，多个key用逗号分隔） | 可选 |
| `SEARXNG_BASE_URLS` | SearXNG 自建实例（无配额兜底，需在 settings.yml 启用 format: json） | 可选 |
| `BING_API_KEYS` | [Bing Web Search v7](https://portal.azure.com/) API Keys（多个用逗号分隔，注：Bing 已宣布退役） | 可选 |
| `GOOGLE_CSE_API_KEYS` + `GOOGLE_CSE_ENGINE_ID` | [Google Programmable Search](https://programmablesearchengine.google.com/) API Key 与引擎 ID（每天 100 次免费） | 可选 |
| `DUCKDUCKGO_ENABLED` | 设为 `true` 启用 DuckDuckGo 免 Key 兜底（推荐作为最后一道防线） | 可选 |
| `NEWS_SEARCH_SOURCE_PRIORITY` | 搜索源优先级，逗号分隔。默认 `bocha,tavily,brave,serpapi,minimax,bing,googlecse,searxng,duckduckgo` | 可选 |
| `TUSHARE_TOKEN` | [Tushare Pro](https://tushare.pro/weborder/#/login?reg=834638 ) Token | 可选 |
| `ENABLE_CHIP_DISTRIBUTION` | 筹码分布开关；开启后优先使用今日缓存，再按 `CHIP_DISTRIBUTION_SOURCE_PRIORITY` 尝试数据源 | 可选 |
| `CHIP_DISTRIBUTION_CACHE_TTL_DAYS` | 筹码分布最近缓存兜底天数，默认 7 | 可选 |
| `CHIP_DISTRIBUTION_SOURCE_PRIORITY` | 筹码分布数据源优先级，默认 `tushare,akshare,instock`；`runStock.sh` 会临时使用 `instock,akshare,tushare` | 可选 |
| `PREFETCH_REALTIME_QUOTES` | 实时行情预取开关：设为 `false` 可禁用全市场预取（默认 `true`） | 可选 |
| `WECHAT_MSG_TYPE` | 企微消息类型，默认 markdown，支持配置 text 类型，发送纯 markdown 文本 | 可选 |
| `NEWS_MAX_AGE_DAYS` | 新闻最大时效（天），默认 3，避免使用过时信息 | 可选 |
| `BIAS_THRESHOLD` | 乖离率阈值（%），默认 5.0，超过提示不追高；强势趋势股自动放宽 | 可选 |
| `AGENT_MODE` | 开启 Agent 策略问股模式（`true`/`false`，默认 false） | 可选 |
| `AGENT_SKILLS` | 激活的策略（逗号分隔），`all` 启用全部 11 个；不配置时默认 4 个，详见 `.env.example` | 可选 |
| `AGENT_MAX_STEPS` | Agent 最大推理步数（默认 10） | 可选 |
| `AGENT_STRATEGY_DIR` | 自定义策略目录（默认内置 `strategies/`） | 可选 |
| `TRADING_DAY_CHECK_ENABLED` | 交易日检查（默认 `true`）：非交易日跳过执行；设为 `false` 或使用 `--force-run` 强制执行 | 可选 |

#### 3. 启用 Actions

`Actions` 标签 → `I understand my workflows, go ahead and enable them`

#### 4. 手动测试

`Actions` → `每日股票分析` → `Run workflow` → `Run workflow`

#### 完成

默认每个**工作日 18:00（北京时间）**自动执行，也可手动触发。默认非交易日（含 A/H/US 节假日）不执行。

> 💡 **关于跳过交易日检查的两种机制：**
> | 机制 | 配置方式 | 生效范围 | 适用场景 |
> |------|----------|----------|----------|
> | `TRADING_DAY_CHECK_ENABLED=false` | 环境变量/Secrets | 全局、长期有效 | 测试环境、长期关闭检查 |
> | `force_run` (UI 勾选) | Actions 手动触发时选择 | 单次运行 | 临时在非交易日执行一次 |
>
> - **环境变量方式**：在 `.env` 或 GitHub Secrets 中设置，影响所有运行方式（定时触发、手动触发、本地运行）
> - **UI 勾选方式**：仅在 GitHub Actions 手动触发时可见，不影响定时任务，适合临时需求

### 方式二：本地运行 / Docker 部署

```bash
# 克隆项目
git clone https://github.com/ZhuLinsen/daily_stock_analysis.git && cd daily_stock_analysis

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env && vim .env

# 运行分析
python main.py
```

> 运行时业务配置以仓库根目录 `.env` 为准；当前 shell 中已有的同名环境变量不会覆盖或补充 `.env`，避免误路由到非预期模型或渠道。

> Docker 部署、定时任务配置请参考 [完整指南](docs/full-guide.md)
> 桌面客户端打包请参考 [桌面端打包说明](docs/desktop-package.md)

## 📱 推送效果

### 决策仪表盘
```
🎯 2026-02-08 决策仪表盘
共分析3只股票 | 🟢买入:0 🟡观望:2 🔴卖出:1

📊 分析结果摘要
⚪ 中钨高新(000657): 观望 | 评分 65 | 看多
⚪ 永鼎股份(600105): 观望 | 评分 48 | 震荡
🟡 新莱应材(300260): 卖出 | 评分 35 | 看空

⚪ 中钨高新 (000657)
📰 重要信息速览
💭 舆情情绪: 市场关注其AI属性与业绩高增长，情绪偏积极，但需消化短期获利盘和主力流出压力。
📊 业绩预期: 基于舆情信息，公司2025年前三季度业绩同比大幅增长，基本面强劲，为股价提供支撑。

🚨 风险警报:

风险点1：2月5日主力资金大幅净卖出3.63亿元，需警惕短期抛压。
风险点2：筹码集中度高达35.15%，表明筹码分散，拉升阻力可能较大。
风险点3：舆情中提及公司历史违规记录及重组相关风险提示，需保持关注。
✨ 利好催化:

利好1：公司被市场定位为AI服务器HDI核心供应商，受益于AI产业发展。
利好2：2025年前三季度扣非净利润同比暴涨407.52%，业绩表现强劲。
📢 最新动态: 【最新消息】舆情显示公司是AI PCB微钻领域龙头，深度绑定全球头部PCB/载板厂。2月5日主力资金净卖出3.63亿元，需关注后续资金流向。

---
生成时间: 18:00
```

### 大盘复盘
```
🎯 2026-01-10 大盘复盘

📊 主要指数
- 上证指数: 3250.12 (🟢+0.85%)
- 深证成指: 10521.36 (🟢+1.02%)
- 创业板指: 2156.78 (🟢+1.35%)

📈 市场概况
上涨: 3920 | 下跌: 1349 | 涨停: 155 | 跌停: 3

🔥 板块表现
领涨: 互联网服务、文化传媒、小金属
领跌: 保险、航空机场、光伏设备
```
## ⚙️ 配置说明

> 📖 完整环境变量、定时任务配置请参考 [完整配置指南](docs/full-guide.md)


## 🖥️ Web 界面

![img.png](sources/fastapi_server.png)

包含完整的配置管理、任务监控和手动分析功能。

**可选密码保护**：在 `.env` 中设置 `ADMIN_AUTH_ENABLED=true` 可启用 Web 登录，首次访问在网页设置初始密码，保护 Settings 中的 API 密钥等敏感配置。详见 [完整指南](docs/full-guide.md)。

### 智能导入

在 **设置 → 基础设置** 中找到「智能导入」区块，支持三种方式添加自选股：

1. **图片**：拖拽或选择自选股截图（如 APP 持仓页、行情列表），Vision AI 自动识别代码+名称，并给出置信度
2. **文件**：上传 CSV 或 Excel (.xlsx)，自动解析代码/名称列
3. **粘贴**：从 Excel 或表格复制后粘贴，点击「解析」即可

**预览与合并**：高置信度默认勾选，中/低置信度需手动勾选；支持按代码去重、清空、全选；仅合并已勾选且解析成功的项。

**配置与限制**：
- 图片需配置 Vision API（`GEMINI_API_KEY`、`ANTHROPIC_API_KEY` 或 `OPENAI_API_KEY` 至少一个）
- 图片：JPG/PNG/WebP/GIF，≤5MB；文件：≤2MB；粘贴文本：≤100KB

**API**：`POST /api/v1/stocks/extract-from-image`（图片）、`POST /api/v1/stocks/parse-import`（文件/粘贴）。详见 [完整指南](docs/full-guide.md)。

**LLM 用量查询**：`GET /api/v1/usage/summary?period=today|month|all`，返回按调用类型和模型分组的 token 消耗汇总（`total_calls`、`total_tokens`、`by_call_type`、`by_model`）。

### 🤖 Agent 策略问股

在 `.env` 中设置 `AGENT_MODE=true` 后启动服务，访问 `/chat` 页面即可开始多轮策略问答。

- **选择策略**：均线金叉、缠论、波浪理论、多头趋势等 11 种内置策略
- **自然语言提问**：如「用缠论分析 600519」，Agent 自动调用实时行情、K线、技术指标、新闻等工具
- **流式进度反馈**：实时展示 AI 思考路径（行情获取 → 技术分析 → 新闻搜索 → 生成结论）
- **多轮对话**：支持追问上下文，会话历史持久化保存
- **导出与发送**：可将会话导出为 .md 文件，或发送到已配置的通知渠道
- **后台执行**：切换页面不中断分析，完成时 Dock 问股图标显示角标
- **Bot 支持**：`/ask <code> [strategy]` 命令触发策略分析
- **自定义策略**：在 `strategies/` 目录下新建 YAML 文件即可添加策略，无需写代码

> **注意**：Agent 模式依赖外部 LLM（Gemini/OpenAI 等），每次对话会产生 API 调用费用。不影响非 Agent 模式（`AGENT_MODE=false` 或未设置）的正常运行。

### 启动方式

1. **启动服务**（默认会自动编译前端）
   ```bash
   python main.py --webui       # 启动 Web 界面 + 执行定时分析
   python main.py --webui-only  # 仅启动 Web 界面
   ```
   启动时会在 `apps/dsa-web` 自动执行 `npm install && npm run build`。
   如需关闭自动构建，设置 `WEBUI_AUTO_BUILD=false`，并改为手动执行：
   ```bash
   cd ./apps/dsa-web
   npm install && npm run build
   cd ../..
   ```

访问 `http://127.0.0.1:8000` 即可使用。

> 也可以使用 `python main.py --serve` (等效命令)

## 🗺️ Roadmap

查看已支持的功能和未来规划：[更新日志](docs/CHANGELOG.md)

> 有建议？欢迎 [提交 Issue](https://github.com/ZhuLinsen/daily_stock_analysis/issues)


---

## ☕ 支持项目

如果本项目对你有帮助，欢迎支持项目的持续维护与迭代，感谢支持 🙏  
赞赏可备注联系方式，祝股市长虹

| 支付宝 (Alipay) | 微信支付 (WeChat) | Ko-fi |
| :---: | :---: | :---: |
| <img src="./sources/alipay.jpg" width="200" alt="Alipay"> | <img src="./sources/wechatpay.jpg" width="200" alt="WeChat Pay"> | <a href="https://ko-fi.com/mumu157" target="_blank"><img src="./sources/ko-fi.png" width="200" alt="Ko-fi"></a> |

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

详见 [贡献指南](docs/CONTRIBUTING.md)

### 本地门禁（建议先跑）

```bash
pip install -r requirements.txt
pip install flake8 pytest
./scripts/ci_gate.sh
```

如修改前端（`apps/dsa-web`）：

```bash
cd apps/dsa-web
npm ci
npm run lint
npm run build
```

## 📄 License
[MIT License](LICENSE) © 2026 ZhuLinsen

如果你在项目中使用或基于本项目进行二次开发，
非常欢迎在 README 或文档中注明来源并附上本仓库链接。
这将有助于项目的持续维护和社区发展。

## 📬 联系与合作
- GitHub Issues：[提交 Issue](https://github.com/ZhuLinsen/daily_stock_analysis/issues)
- 合作邮箱：zhuls345@gmail.com

## ⭐ Star History
**如果觉得有用，请给个 ⭐ Star 支持一下！**

<a href="https://star-history.com/#ZhuLinsen/daily_stock_analysis&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ZhuLinsen/daily_stock_analysis&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ZhuLinsen/daily_stock_analysis&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ZhuLinsen/daily_stock_analysis&type=Date" />
 </picture>
</a>

## ⚠️ 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。作者不对使用本项目产生的任何损失负责。

---
