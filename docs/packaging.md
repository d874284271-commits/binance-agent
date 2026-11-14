# 构建与分发未签名预览版

在同一源码提交上构建 `0.3.0a1`，得到 Windows x64 ZIP 与 macOS Apple Silicon arm64 `.app` ZIP。CI 手动运行会在两平台成功后汇总对应源码、依赖清单及 `SHA256SUMS`；普通 push/tag 不创建 Release。

```bash
uv sync --locked --extra desktop --extra build
uv run --no-sync pyinstaller packaging/binance-agent.spec --noconfirm
```

在目标平台原生构建，统一使用 Python 3.12。不要从已有 `.venv` 的任意依赖版本直接打包；`uv sync --locked` 会移除过时依赖，spec 再校验实际收集的发行版本与锁文件。

| 目标 | 原始产物 | 候选归档 |
| --- | --- | --- |
| Windows x64 | `dist/binance-agent/`，入口 `binance-agent.exe` | `binance-agent-0.3.0a1-windows-x64-unsigned.zip` |
| macOS arm64 | `dist/Binance Agent.app` | `binance-agent-0.3.0a1-macos-arm64-unsigned.zip` |

macOS 构建限定 arm64；不宣称支持 Intel Mac、Windows ARM 或尚未原生验收的系统版本。CI 使用 `macos-15` arm64 和 Windows x64 runner，runner 构建成功不是最低系统版本或桌面交互验收。

## 下载后使用

先核对维护者公布的下载来源、版本和 `SHA256SUMS`。Windows 必须完整解压整个 onedir 文件夹，不能只移动 exe；安装 Microsoft [WebView2 Evergreen Runtime](https://developer.microsoft.com/microsoft-edge/webview2/) 后启动。macOS 解压后将 `.app` 放到用户选择的应用目录再启动。

本次是未签名、未公证预览版。Windows 可能显示未知发布者提示；macOS 可能阻止启动。只有确认来源与校验值后，才使用系统针对该应用提供的继续打开/允许选项；如系统不提供例外入口，请停止并报告具体提示，不关闭全局安全防护、不批量移除隔离属性。macOS 构建中的 ad-hoc 签名没有发行者身份，不等同于 Developer ID 签名或公证。

更新时先退出应用、备份用户数据，再替换完整应用目录；不要把旧版本文件叠加到新包中。卸载只删除应用本身，用户数据保留。预览版的原生安装、三种研究入口、保存、停止、关闭/重开仍需在最终候选提交上验证。

## 依赖与许可

- `feedparser==6.0.14` 使用 `feedparser-sgmllib==2.1.0`，旧 `sgmllib3k` 不再进入锁文件或构建环境。
- PyObjC core 与相关 framework 统一为 `12.2.2`。具体依赖与下载哈希以 `uv.lock` 为准；索引显式固定到官方 PyPI，不依赖开发机镜像配置。
- spec 保留项目 MIT、Python 许可、实际打包发行包的元数据及随附许可，并加入 `packaging/licenses/` 的上游正文。
- RSS 两份许可从相应版本 PyPI wheel 原样提取；PyObjC 正文与 `12.2.2` sdist 对照。来源、版本、归档内路径及下载包 SHA-256 见 [sources.json](../packaging/licenses/sources.json)。
- `bundle-build.json` 保存源码版本、提交、锁文件哈希及实际收集的依赖版本；不保存用户配置、环境变量或凭证。

这是对明确旧缺口的修复；最终两平台包仍须复核实际清单和第三方文件，不能把依赖元数据复制步骤视为所有非 Python 文件均已完成许可审查。

## 本地候选验证与归档

Windows PowerShell：

```powershell
uv run --no-sync python packaging/release.py check --target windows-x64
pwsh -NoProfile -File packaging/windows-smoke.ps1
uv run --no-sync python packaging/release.py package --target windows-x64 --output candidate-artifacts
```

macOS：

```bash
uv run --no-sync python packaging/release.py check --target macos-arm64
uv run --no-sync python packaging/macos-smoke.py
uv run --no-sync python packaging/release.py package --target macos-arm64 --output candidate-artifacts
```

两种 smoke 都禁用 `.env` 自动加载，使用独立数据与配置、随机本机端口和合成 `demo`。先校验版本/profile、health 及 `/reports/example` 的合成标记，再通过 `/research/sessions` 分别创建市场研报与独立专题并启动 initial 轮次；轮询到该 run 完成后校验 `/runs/{id}/report`、引用原文和 Markdown 导出。独立专题必须没有资产范围或交易建议。随后清理启动的进程，诊断与两份报告保留在 `smoke-artifacts/`。这不使用已移除的评估/回放/观察路由，也不调用真实模型；SSE 实时更新、WebView2/原生窗口、剪贴板和保存对话框另由最终原生验收覆盖。

不要将 `candidate-artifacts/`、`smoke-artifacts/` 或汇总输出加入源码公开清单。spec 允许本地调试构建并标记 `source_dirty`；正式候选汇总拒绝未提交源码、缺失平台、混合版本/提交/锁文件或校验值不符的归档。

## CI 与同步交付

`.github/workflows/ci.yml` 保留现有检查矩阵。仅 `workflow_dispatch` 在检查通过后构建两平台。可在 Actions 的 Run workflow 中填入 `commit`（完整 40 位 SHA）；留空则使用所选分支/标签触发时的提交。检查、两平台构建和汇总的 checkout 均固定到同一提交，拒绝不匹配的 checkout。工作流只需 `contents: read`，没有 `gh release` 或上传 Release 的步骤。

候选提交到仓库后，也可明确指定该 SHA 触发并下载完整汇总 ZIP：

```bash
gh workflow run ci.yml --ref <包含该工作流的分支> -f commit=<完整候选提交SHA>
gh run list --workflow ci.yml --event workflow_dispatch
gh run download <run-id> --name unsigned-preview-<完整候选提交SHA> --dir preview-download
```

网页端在对应运行的 Artifacts 下载 `unsigned-preview-<SHA>` ZIP。没有提交的本地修改不会进入远端候选；主负责人先整合候选提交，再触发工作流并将 Windows ZIP 用于已连接远程桌面的原生验收。

两平台产物齐全后，汇总任务运行：

```bash
python packaging/release.py aggregate --incoming incoming --output preview-candidate
```

结果包含两份 unsigned ZIP、各平台 JSON 清单、该提交的 Git 源码 ZIP、`candidate.json` 与 `SHA256SUMS`。macOS ZIP 使用 `ditto` 保留 `.app` 所需的文件结构与链接。源代码归档来自已提交 Git 树，不会带入工作区运行数据。

后续由主负责人核对原生验收、公开内容与最终说明，再使用这些完整材料准备私有仓库 draft，最后同步公开源码并发布 prerelease；构建成功不会代替该阶段。版本唯一来源为 `pyproject.toml`，打包时要求 `src/binance_agent/__init__.py` 同步；平台资源与文件名从该版本生成。

## 配置、数据与诊断

| 内容 | 源码运行 | 冻结包 |
| --- | --- | --- |
| 研究记录 `.runs/`、SQLite checkpoint | `<CWD>/.runs/` | `~/.binance_agent/.runs/` |
| 日志 | stderr | stderr + `~/.binance_agent/logs/agent.log` |
| 密钥库及运行时配置 | `~/.binance_agent/` | 相同用户目录 |

`BINANCE_AGENT_DATA_DIR` 可隔离运行记录；`BINANCE_AGENT_RUNTIME_CONFIG` 可指定运行时配置。`config.toml` 优先级为 `BINANCE_AGENT_CONFIG` → 当前工作目录的 `config/config.toml` → 可执行文件旁的配置 → 包内默认配置 → 开发仓库配置。

双击无反应时查看应用日志并先脱敏再报告；端口占用时在自定义配置修改 `[app] port`。排查可用 `BINANCE_AGENT_HEADLESS=1`，这只是诊断用后端运行方式。不要用真实数据目录运行 smoke，不要随包携带 `.env`、`secrets.enc`、运行时覆盖或个人研究记录。
