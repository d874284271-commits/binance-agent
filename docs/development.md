# 开发与验证

先运行完全离线的合成研究服务，再打开本地页面检查交互：

```bash
uv sync --frozen --extra dev --extra desktop
uv run --frozen python -m binance_agent.research_demo
# http://127.0.0.1:8766
```

该演示通过固定模型和合成材料覆盖研究、核验、会话与报告，不证明真实模型效果。使用独立的 `BINANCE_AGENT_DATA_DIR` 与 `BINANCE_AGENT_RUNTIME_CONFIG` 进行验收，避免污染日常数据。

## 结构与责任

- `app/research_sessions.py` 处理幂等会话命令、轮次、后台执行、预算恢复及 SSE。
- `app/collaboration_store.py` 是会话、消息、结果和输入版本的 SQLite 权威；与 checkpoint 共库，版本迁移先备份再事务提交。
- `agents/graph.py` 按研究意图选择市场或专题路径。市场保留预检、分析、风控与最终证据闸门；专题进入 `agents/topic.py`，复用 `agents/research.py` 的正文研究与核验。
- `agents/reports.py` 定义统一报告和 Markdown 转换，`app/reports.py` 只读取保存的结果，不触发模型。
- `intel/aggregator.py` 按问题和可选资产发现资料；独立专题仅使用明确支持公共/话题资料的提供方。`intel/search.py` 只做可选 Tavily 检索，正文仍由原抓取链读取。
- `desktop/ui` 是静态 HTML/CSS/JS，无构建步骤；`desktop/shell.py` 负责窗口、原生保存与后端关闭。

对外会话契约与兼容规则见[发布计划](release-plan.md)和[持续研究说明](research/collaboration.md)。旧客户端保留原 symbol/question 契约，旧记录不猜测为新产品意图。

## 检查

```bash
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy src
node --check src/binance_agent/desktop/ui/app.js
uv run --frozen pytest
```

改动行为后优先运行对应测试；候选冻结后运行完整检查。浏览器测试不能替代 Windows WebView2 和 macOS 原生流程，打包与实际平台验收见[打包文档](packaging.md)。

## 开发者实验

评估、校准与重放代码保留在 `eval`、`app/run_replay.py` 等源码模块中。通过 `python -m binance_agent.eval --help` 查看明确支持的实验命令；这些能力不再暴露为产品导航或 HTTP 评估/校准/观察探测/回放接口。

历史实验账本不随本次源码分发。公开源码不包含个人模型配置、逐轮运行数据、密钥、工具会话数据库或未经许可的网页正文。实验数据要分别核查来源与再分发依据；合成 fixture 可用于行为回归。
