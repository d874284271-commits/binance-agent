"""Production UI remains a static, build-free product surface."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_static_assets_parse_without_a_frontend_build():
    ui = Path("src/binance_agent/desktop/ui")
    html = (ui / "index.html").read_text()
    assert "/ui/app.js" in html and "/ui/styles.css" in html
    assert "<script>" not in html
    executable = shutil.which("node")
    if not executable:
        pytest.skip("Node required")
    subprocess.run([executable, "--check", str(ui / "app.js")], check=True)
