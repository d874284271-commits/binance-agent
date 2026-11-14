# Changelog

本项目的重要变更记录在此文件中。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本遵循语义化版本约定。

## [0.3.0a1] - Unreleased

本条目描述当前开源预览候选，发布及平台验收以同版本 Release 说明为准。

- 新增一键市场研报、关联币种专题及无需交易对的独立专题。
- 新增统一保存的报告、版本回看、复制与 Markdown 原生保存；导出不调用模型。
- 会话记录研究意图和范围；更新、追问、补充与恢复保留各自语义和预算约束。
- 专题复用正文研究与核验，独立专题不请求 Binance 行情；搜索可选接入 Tavily，失败保留已有证据。
- SQLite 迁移前备份并使用版本化事务，兼容旧已完成记录。
- 界面收敛为研报、历史、设置，首次提供合成示例；默认关闭热键和置顶，默认窗口 1040×760。
- 从产品界面和 HTTP 路由移除调试、评估校准、伪提醒与调试回放；开发者实验代码保留。
- 版本统一为未签名、未公证的 Windows x64 / macOS arm64 开源预览候选。

## [0.2.0] - Unreleased

以下保留 v0.2.0 开发阶段的历史变更。当前条目只描述仓库中的候选变更，不表示已经完成 tag、push 或远端 Release。

### Added

- 在任何分析师 LLM fan-out 前执行确定性 `data_preflight`：按缺失类别补抓行情或资讯，每类最多一次；仍阻断时直接生成 `no_trade`。
- `preflight_refetch_enabled` 配置，以及旧 `strategy_planner_refetch_enabled` 的单版本弃用兼容。
- 逐模型 `RuntimeModelCandidate`，分别携带 endpoint、model、timeout、retry 与 `structured_output` 能力。
- 结构化辩论证据契约：`source_path` + JSON 标量 `source_value`。
- 证据完整性报告：`verified_ratio` 以及 verified、missing、mismatch、invalid 计数。
- 辩论证据按独立数据点计量：`by_side`、`duplicate_claims`、`shared_sources`、逐轮 `rounds` 与 `convergence`。
- 辩论在开篇轮之后，若上一轮没有新增已核验证据即提前收敛（`stop_reason=no_new_evidence`）；辩手输入新增 `previously_cited_paths`。
- `AnalystReport.data_sources`，由节点代码按实际输入填写，不来自 LLM。
- 不可信新闻标题数据区 `untrusted_headlines`，配套 HTML/控制字符清洗、长度限制和 prompt-injection 防护。
- 本地 Host/Origin/`Sec-Fetch-Site` 守卫、默认 2 个分析任务并发限制和代理 URL 凭证校验。
- Outcome 的评估起止时间、实际模拟成交价、被评估目标和 `policy_version`。
- `decisive_brier`、删失统计、每桶 Wilson 95% 区间和可审计 calibration proposal。
- Eval CLI `--proposal-out PATH`，只导出可审计 JSON，不自动应用校准。
- `confidence.bucket_caps` 与需要人工二次确认的 `POST /config/calibration/apply`。
- `default`、`demo`、`demo-live` 三个运行 profile。
- BTCUSDT、ETHUSDT、SOLUSDT 三个完全离线且确定性的演示场景。
- UI“评估”视图、preflight/证据完整性/token usage trace，以及无需构建步骤的独立 `styles.css` / `app.js`。
- 可编辑 draw.io 架构图、MIT License 和 v0.2.0 发布候选说明。
- 分析视图与评估视图的真实离线 demo 截图。

### Changed

- 后置 strategy planner 不再补抓行情或资讯，只负责受限流程决策。
- RunStore、checkpoint 与 replay 使用 preflight 提升后的同一 canonical snapshot。
- LLM fallback 使用候选自身的结构化输出能力，不再继承主端点 JSON mode。
- LLM Registry 共享代理 HTTP client，并在代理切换和应用关闭时显式关闭。
- `debate_grounding` 保留外层兼容，但主指标更名为 `verified_ratio`；`grounded_ratio` 仅作临时别名。
- 分析师方向共识按独立数据源计票，同源分析师合并为一票；情绪分析师不再消费资金费率、多空比和标记价。
- 有方向建议的辩论证据 cap 改看支持方的独立证据，反方可核验证据更多时也会限制置信度；research_manager 与 strategy_synth 不再按 claim 自报 confidence 加权。
- Outcome 从 `completed_at` 起算，只使用窗口内完整收盘 K 线，并采用 `ohlc-conservative-v1` 保守裁决。
- Confidence 先计算现有确定性规则下的 `pre_policy_confidence`，再应用人工批准的分桶 cap。
- README 统一使用“证据完整性”，明确项目不判断论点语义真伪，也不把 Outcome 称为真实绩效。
- 默认 LLM 说明与 `config/config.toml` 对齐为 `https://api.openai.com/v1`、`gpt-4o-mini` 和 `OPENAI_API_KEY`。

### Fixed

- 修复 preflight 补抓后分析师、checkpoint、RunStore 和 replay 读取不同快照的问题。
- 修复主端点与 fallback 端点结构化输出能力不一致时的错误参数继承。
- 修复 endpoint 更新、UI 修改和持久化重启后 `structured_output` 丢失的问题。
- 修复容器引用、空值、缺失路径和不匹配标量被错误视为有效证据的问题。
- 修复 Outcome 回退到 `created_at`、包含部分 K 线和超过单页限制时截断的问题。
- 修复 long/short 入场模拟过于乐观，以及同根止损/止盈冲突时结果不保守的问题。
- 修复把未入场、持有中或无法评估样本混入 Brier 与命中率口径的问题。
- 修复过期或数据集已变化的 calibration proposal 仍可能被应用的问题。

### Security

- 应用 Host 仅允许 loopback；受保护路由拒绝非本地 Origin、跨站 `Sec-Fetch-Site` 和非 loopback 无浏览器元数据请求。
- 新闻标题视为不可信外部数据，不允许覆盖 system prompt 或触发工具行为。
- 运行时代理 URL 禁止 userinfo；认证代理只能通过环境变量注入，日志不输出凭证或完整地址。
- 分析并发超限返回 HTTP `429`，避免本地页面或意外请求无限触发昂贵调用。

### Compatibility

- 历史 `source_ref=path=value` 仅保留读取兼容，新 prompt 不再生成。
- `debate_grounding.grounded_ratio` 暂时映射到 `verified_ratio`。
- `strategy_planner_refetch_enabled` 暂时映射到 `preflight_refetch_enabled` 并发出弃用提示。
- 运行记录和 Outcome 保持向后可读；缺少 `completed_at` 的历史 run 明确记为 `unevaluable`。

### Out of scope

- 不增加下单、仓位、杠杆、账户或真实交易执行能力。
- Outcome 不模拟手续费、滑点、多目标、仓位管理或资金费率成本。
- 校准不会自动应用，也不宣称自动学习或自我进化。

## [0.1.0]

- 初始本地多智能体市场分析、FastAPI/SSE、桌面悬浮窗、Binance 行情、资讯聚合、LangGraph 编排、运行时端点配置、checkpoint、RunStore 与基础 Outcome 观测能力。
