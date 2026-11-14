# 持续研究会话

在“研报”选择市场研报或指定话题。市场研报只需币种和市场；话题输入问题，可独立研究或关联币种。“更新当前研报 / 话题”和“继续追问”都创建新轮次；“运行中补充”写入当前轮。新建研究、改变入口或研究对象时创建新会话。

离线操作演示：

```bash
uv run --frozen python -m binance_agent.research_demo
# 浏览器打开 http://127.0.0.1:8766
```

该演示使用合成行情、正文和模型响应，包含 BTC 正常正文、ETH 标题误导、SOL 正文缺失三种场景，不调用真实模型。另有不关联币种的独立专题。页面展示结论、依据、来源、分析时点及补充处理状态；内部执行事件保留在本地存储，不作为产品导航。

## 页面与进程生命周期

- 关闭页面或 SSE 断线：后台继续运行；重新打开页面恢复已有会话和事件订阅，不重新发起研究。
- 停止本轮：取消执行并释放并发槽，保留历史和会话；下一轮重新计预算。
- 退出后端或整个桌面应用：取消活动执行，持久化为 `interrupted`。下次启动只展示中断状态，用户点击恢复才继续请求。
- 恢复：保留原快照、已用预算、未知调用费用和绝对截止时间，要求配置指纹一致；超时或预算耗尽时给出保守结果。
- 关闭会话：先停止未结束的运行，再关闭；保留历史，不能继续接收任务。

Agent 的任务状态和驻留状态分开保存。默认空闲 60 秒卸载上下文，驻留上限 32；正在调用模型或工具的 Agent 不会被卸载。共享模型客户端和连接池随应用关闭。

## 补充信息的含义

“已接收”表示数据库已保存，不代表模型已经处理。当前调用完成后，在阶段边界将补充纳入 checkpoint，并以新输入版本重新执行分析、汇总、辩论、合成和原有风险闸门。已采集的本轮原文和冻结行情可以复用，请求数、截止时间、文档尝试与任务调查轮数不会重置。调查轮次按任务、输入版本和轮次幂等预留，恢复同一 checkpoint 不会再次扣减。

旧版本步骤保留审计记录，但不会进入最新结果视图。最终提交在事务内核对版本和用户消息状态；若补充尚未纳入，则不能发布旧版结果。额度不足时市场研报返回保守建议，专题返回 `insufficient_evidence` 报告并记录未处理事项，不伪造交易建议。已完成运行拒绝补充，应使用“继续追问”或“更新”创建新轮。

历史结论以来源运行和分析时点标记。它们是历史材料；新一轮仍重新取证，不会把旧正文自动视为当前事实。

## 本机 API

所有写操作适用现有 loopback/Origin 保护。调用示例：

```bash
curl -s http://127.0.0.1:8765/research/sessions \
  -H 'Content-Type: application/json' \
  -d '{"mode":"topic","target":{"kind":"independent"},"question":"数字资产监管提案有哪些未解决事项？","request_id":"session-1"}'
# 取响应的 session_id，替换下面的 SESSION。
curl -s http://127.0.0.1:8765/research/sessions/SESSION/turns \
  -H 'Content-Type: application/json' \
  -d '{"action":"initial","request_id":"turn-1"}'
# 取响应的 run_id，替换 RUN。
curl -s http://127.0.0.1:8765/research/sessions/SESSION/messages \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"RUN","content":"区分申请与批准","request_id":"message-1"}'
curl -N 'http://127.0.0.1:8765/research/sessions/SESSION/events?after=0'
```

| 接口 | 行为 |
|---|---|
| `POST /research/sessions` | 创建固定 mode/target 和原始目标的会话；不启动模型 |
| `GET /research/sessions[/{id}]` | 列表或详情，包含轮次、Agent、任务、补充处理状态 |
| `POST /research/sessions/{id}/turns` | initial/update/followup 新轮；同一会话只允许一个未结束运行 |
| `POST /research/sessions/{id}/messages` | 活动运行补充；必须指定预期 `run_id` |
| `GET /research/sessions/{id}/events` | 持久化 SSE；支持 `after` 和 `Last-Event-ID` |
| `POST /runs/{id}/resume`、`/stop` | 手动恢复中断轮次或停止本轮 |
| `POST /research/sessions/{id}/close` | 关闭空闲会话 |
| `GET /runs/{id}/report[.md]` | 当前指定已保存版本的报告或 Markdown，不调用模型 |

创建和控制请求使用 `request_id` 幂等；重试应复用原 ID，相同 ID 配不同内容返回 409。补充返回 202。新问题和补充各支持最多 2000 字符，超长返回校验错误，不静默截断。并发满返回 429；旧页面补充、重复恢复、配置不一致等状态冲突返回 409。活动协作期间配置写入返回 409。

兼容的 `GET /analyze` 和 `GET /runs/{id}/resume` 对协作运行使用同一后台管理器。旧 JSONL 已完成结果显示为“历史研究”，不猜测意图、不伪造会话或 Agent 历史。不兼容的旧中断记录需新建研究。

## 配置与数据

```toml
[runtime]
checkpoint_backend = "sqlite"
checkpoint_path = ".runs/checkpoints.sqlite"

[collaboration]
enabled = true
idle_unload_s = 60
max_resident_agents = 32
```

同时启用 `[research] enabled = true` 并设置 `max_requests`、`deadline_s`。关闭协作开关可使用原单轮入口。协作模式要求 SQLite，不能配 `memory`。

会话、角色任务、消息、事件、预算、结果版本和 checkpoint 保存在同一个 SQLite 控制库；证据正文仍按现有证据存储读取。新协作运行不双写 JSONL。旧 checkpoint 整体快照在事务中迁移到按线程/namespace/channel 存储的表，保留旧表和迁移标记，迁移失败回滚并保留原数据。活动运行和开放会话恢复所需的线程被保护，不受历史清理影响。

开发态相对路径挂在当前工作目录，冻结包挂在 `~/.binance_agent/`；`BINANCE_AGENT_DATA_DIR` 可覆盖数据根目录，demo 额外使用 `.demo/` 子目录。SQLite、证据和用户配置应一起保留。桌面 WebView 使用应用数据目录内的 `.desktop/` 持久配置，保留会话选择和事件游标；demo 同样隔离到 `.demo/`。

外部模型请求完成到结果 checkpoint 之间仍有崩溃窗口；恢复可能再次请求，不承诺外部调用 exactly-once。内部消息以 checkpoint 游标和幂等 outbox 投递保持可恢复。架构验证不代表真实模型质量、收益或耗时改善。

## 补充后的完成含义

“已进入本轮上下文”（API `incorporated`）表示消息已纳入当前输入版本。是否完成调查仍看任务结果、缺口和最终结果。新问题会复用缓存正文并重新建立分析覆盖；预算不足或期限已到时明确保留未完成事项。修订前的中断 Run 因研究配置指纹改变不能直接恢复，需新建一轮；已完成历史仍可查看。

报告完成与消息版本核对在同一 SQLite 事务中提交。独立专题的 symbol/market 为 NULL；研究模式和范围在会话创建后保持不变。schema v1 迁移前创建 SQLite 一致性备份，失败回滚，不重建空数据库。报告完整性 complete/partial/insufficient_evidence 与运行状态分开，历史轮次的保存结果不会因后续更新而变化。
