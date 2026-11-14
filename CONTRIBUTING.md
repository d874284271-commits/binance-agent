# 参与开发

修复解析或桌面问题时，先在离线示例中复现，再运行相关测试：

```bash
uv sync --locked --extra dev --extra desktop --extra build
uv run --no-sync pytest tests/test_rss_provider.py tests/test_windows_smoke.py
```

使用 Python 3.10–3.12；候选构建统一 Python 3.12。`uv.lock` 是两平台共同依赖基线，不手改版本或下载哈希；确有依赖调整时更新 `pyproject.toml`，运行 `uv lock`，一并检查许可及受影响行为。桌面打包与候选汇总见 [打包说明](docs/packaging.md)。

## 提交问题与修改

- 问题报告写明应用版本、系统/架构、复现步骤、实际与预期结果；优先附合成输入。
- PR 说明用户可观察的改变、必要的兼容处理及实际验证结果。保持修改围绕一个问题，保留已有用户配置和历史数据。
- 对行为变化补最小必要回归；仅运行相关检查，候选冻结时由 CI 执行完整检查。
- 使用项目现有 Ruff、mypy、pytest 和原生 HTML/CSS/JS，不为小改动新增框架或构建系统。

```bash
uv run --no-sync ruff check src tests packaging
uv run --no-sync ruff format --check src tests packaging
uv run --no-sync mypy src
node --check src/binance_agent/desktop/ui/app.js
```

UI 变化需实际检查宽窄窗口、键盘操作、历史和证据阅读。PowerShell 控制流测试、headless smoke 均不能代替 Windows WebView2 或 macOS 原生窗口验收。

## 数据与安全

测试使用 `demo` 和独立临时数据目录。不要提交 `.env`、密钥、运行时配置、个人研究记录、原始服务日志、带认证信息的 URL 或用户截图。需要实验数据时优先使用合成资料；引入真实网页或第三方素材时说明来源、范围和再分发依据。项目 MIT 许可不自动覆盖这些材料。

漏洞及疑似凭证泄露按 [SECURITY.md](SECURITY.md) 处理，不放入公开 issue。当前版本仅分析，不接交易执行；回归和演示不应触发真实下单或未经明确安排的付费模型调用。

## 发布边界

`0.3.0a1` 是未签名、未公证的预览版本。普通 push/tag 只运行检查；手动工作流构建两平台候选并汇总源码与校验文件，不自动创建 GitHub Release。维护者在原生验收、公开文件与许可检查完成后另行准备 draft/prerelease，不把单个平台产物当作完整发行。
