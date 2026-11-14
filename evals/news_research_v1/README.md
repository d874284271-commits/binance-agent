# news_research_v1 离线评测资料

`cases.json` 包含 40 个**虚构合成**回归案例：24 个 `dev`、16 个 `holdout`。
它们用于检查标题、摘要和正文证据链是否保留否定、条件、数字、时间、资产范围与
未解决状态；不对应真实新闻、真实行情、投资建议或模型效果。每个 `event_id` 只出现
在一个 split，防止同一事件跨开发集和保留集泄漏。

`search_queries.json` 是 Tavily/Brave 的 12 条候选查询和手工结果记录格式。该资料
不含密钥、不发送请求、不比较真实价格，也不选供应商赢家。若未来有明确授权和凭证，
应以同一查询记录前五项的官方来源覆盖、相关性、费用和时间元数据；搜索摘要只能作为
发现线索，不能作为正文证据。

## 两个独立对照

使用 `binance_agent.eval.research_eval.build_comparison_manifests` 生成：

1. 内容对照固定 `multi_no_debate` 工作流，只变更 `title`、`summary`、`evidence` 输入。
2. 工作流对照固定 `evidence` 输入，只变更 `single`、`multi_no_debate`、`multi_debate`。

两组清单不得混合聚合，否则无法知道差异来自内容还是工作流。每份 manifest 记录冻结
输入指纹与模型配置；本模块不会运行真实模型，也不管理费用或预算。

JSONL runner 每行至少给出 `case_id` 和 `output`，可选 `status`、`token`、`latency_ms`、
`cost_usd`。评分报告把缺失与 `unsupported` 留在分母中，未知资源指标保持 `null` 并单独
计数。评分是词项和 fixture 覆盖的工程检查；人工核对原始来源、事实性判断和真实模型
效果实验仍待完成，不能把通过率表述为实验成功。
