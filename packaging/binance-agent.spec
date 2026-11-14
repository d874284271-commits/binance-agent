# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置(onedir):桌面悬浮窗 + 内置 UI 静态资源 + 默认配置。

构建(需先 ``uv sync --locked --extra desktop --extra build``):
    uv run pyinstaller packaging/binance-agent.spec --noconfirm
产物: dist/binance-agent/binance-agent.exe
运行要求: Windows 10/11 自带的 WebView2 Runtime(pywebview EdgeChromium 后端)。
数据目录: 冻结态统一写 ~/.binance_agent/(.runs、logs、secrets)。
"""

import json
import sys
import sysconfig
from importlib.metadata import packages_distributions
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata
from PyInstaller.config import CONF

ROOT = Path(SPECPATH).parent  # noqa: F821 — SPECPATH 由 PyInstaller 注入
sys.path.insert(0, str(ROOT / "packaging"))
from release import source_identity, verify_dependencies

identity = source_identity(ROOT)
VERSION = identity["version"]

datas = [
    # 悬浮窗单页 UI(server.py 经 __file__ 相对路径定位,冻结后同样位于包内)
    (str(ROOT / "src" / "binance_agent" / "desktop" / "ui"), "binance_agent/desktop/ui"),
    # 默认配置(settings.default_config_path 的冻结回退路径)
    (str(ROOT / "config" / "config.toml"), "config"),
    # 分发本项目代码时一并保留 MIT 版权与许可文本。
    (str(ROOT / "LICENSE"), "."),
    # 少数上游 wheel 未附完整许可文本;来源记录与补充正文一并分发。
    (str(ROOT / "packaging" / "licenses"), "third-party-licenses"),
]

python_license = next(
    path
    for root in (Path(sys.base_prefix), Path(sysconfig.get_path("stdlib")))
    for path in (root / "LICENSE.txt", root / "LICENSE")
    if path.is_file()
)
datas.append((str(python_license), "third-party-licenses/python"))

# Main-text extraction uses packaged settings, stoplists and URL metadata.
datas += (
    collect_data_files("trafilatura")
    + collect_data_files("justext")
    + collect_data_files("courlan")
)

hiddenimports = [
    # feedparser 6.0.14 使用有完整许可的独立 SGML parser。
    "feedparser_sgmllib",
    # uvicorn 事件循环/协议按字符串动态导入
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
    # anyio 后端动态导入(starlette/httpx 依赖)
    "anyio._backends._asyncio",
    # pywebview Windows 渲染后端(EdgeChromium/WinForms)运行时选择
    *collect_submodules("webview.platforms"),
]
if sys.platform == "win32":
    hiddenimports += ["keyring.backends.Windows", "pynput.keyboard._win32", "pynput.mouse._win32"]
elif sys.platform == "darwin":
    hiddenimports += ["keyring.backends.macOS", "pynput.keyboard._darwin", "pynput.mouse._darwin"]

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # 排除测试与可选大件(akshare/pandas 属 intel-cn 可选组,不进默认包)
    excludes=["tkinter", "pytest", "pandas", "matplotlib", "akshare", "sgmllib"],
    noarchive=False,
)

# PyInstaller 不会自动保留每个依赖的版权文件。按实际收集到的模块补齐发行
# 元数据(包含 wheel 自带的 licenses/),不复制开发环境里未打包的依赖。
distribution_map = packages_distributions()
bundled_distributions = {
    distribution
    for name, _, _ in [*a.pure, *a.binaries]
    for distribution in distribution_map.get(name.split(".")[0].split("/")[0], [])
}
actual_versions = verify_dependencies(bundled_distributions, ROOT)
for distribution in sorted(bundled_distributions):
    for source, destination in copy_metadata(distribution):
        source_path = Path(source)
        for item in source_path.rglob("*"):
            if item.is_file():
                target = str(Path(destination) / item.relative_to(source_path))
                a.datas.append((target, str(item), "DATA"))

manifest = Path(CONF["workpath"]) / "bundle-build.json"
manifest.write_text(
    json.dumps({**identity, "dependencies": actual_versions}, indent=2) + "\n", encoding="utf-8"
)
a.datas.append(("bundle-build.json", str(manifest), "DATA"))

platform_options = {}
if sys.platform == "win32":
    from packaging.version import Version
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo,
        StringFileInfo,
        StringStruct,
        StringTable,
        VarFileInfo,
        VarStruct,
        VSVersionInfo,
    )

    parsed_version = Version(VERSION)
    numeric_version = (*parsed_version.release, parsed_version.pre[1] if parsed_version.pre else 0)
    platform_options["version"] = VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=numeric_version,
            prodvers=numeric_version,
            mask=0x3F,
            flags=0x2 if parsed_version.is_prerelease else 0,
            OS=0x40004,
            fileType=0x1,
            subtype=0x0,
            date=(0, 0),
        ),
        kids=[
            StringFileInfo(
                [
                    StringTable(
                        "040904B0",
                        [
                            StringStruct("FileDescription", "Binance Agent unsigned preview"),
                            StringStruct("FileVersion", VERSION),
                            StringStruct("ProductName", "Binance Agent"),
                            StringStruct("ProductVersion", VERSION),
                            StringStruct("OriginalFilename", "binance-agent.exe"),
                            StringStruct(
                                "LegalCopyright", "Copyright (c) 2026 Binance Agent contributors"
                            ),
                        ],
                    )
                ]
            ),
            VarFileInfo([VarStruct("Translation", [1033, 1200])]),
        ],
    )
elif sys.platform == "darwin":
    platform_options.update(target_arch="arm64", codesign_identity=None)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="binance-agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # 悬浮窗应用不带控制台;日志见 ~/.binance_agent/logs/agent.log
    disable_windowed_traceback=False,
    **platform_options,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="binance-agent",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Binance Agent.app",
        bundle_identifier="io.binance-agent.desktop",
        version=VERSION,
        info_plist={
            "CFBundleVersion": VERSION,
            "NSHumanReadableCopyright": "Copyright (c) 2026 Binance Agent contributors",
        },
    )
