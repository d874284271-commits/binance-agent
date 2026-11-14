# 研究问题排查

报告没有足够依据时，先展开“来源与缺口”，确认哪些资料取得成功、哪些仍待核实。运行被中断时查看页面错误原因；确认模型配置后可手动恢复，原预算与截止时间不会重置。

| 症状 | 排查位置 |
| --- | --- |
| 无法生成研报 | 设置中的模型连接状态、当前错误原因 |
| 某来源不可用 | 报告来源与缺口，内部 intel_snapshot.source_status |
| 运行超时或预算耗尽 | 保存的 research_budget 与本轮未完成事项 |
| 恢复提示配置不一致 | 使用当前配置新建一轮，原历史保留 |
| 打包后无法启动 | 应用数据目录内 logs/agent.log |

## 开发者诊断

```bash
curl http://127.0.0.1:8765/health
curl 'http://127.0.0.1:8765/research/sessions'
curl 'http://127.0.0.1:8765/runs?limit=10'
# 将 RUN 替换为实际 run_id
curl http://127.0.0.1:8765/runs/RUN/report
```

协作会话、轮次、事件、预算和 checkpoint 保存在 SQLite 控制库，旧版 JSONL 仍可读取。开发态路径基于当前工作目录，打包态基于 `~/.binance_agent/`；`BINANCE_AGENT_DATA_DIR` 可覆盖，demo 另行隔离。位置与恢复约束见[会话文档](research/collaboration.md)。

内部事件用于定位模型、工具和资料失败，产品页面只保留用户可采取行动的信息。实验校准、观察探测和调试重放 HTTP 路由已移除；源码开发工具见[开发说明](development.md)。

提交问题前只提供脱敏的错误原因和合成复现，删除密钥、认证头、带凭证 URL、用户名及用户研究正文。不要公开整个运行数据库或配置目录。
