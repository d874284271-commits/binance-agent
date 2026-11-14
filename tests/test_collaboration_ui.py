"""Product navigation and session control contract; detailed behavior in product tests."""

from pathlib import Path

UI = Path("src/binance_agent/desktop/ui")


def test_product_controls_keep_research_and_remove_engineering_surfaces():
    html = (UI / "index.html").read_text()
    js = (UI / "app.js").read_text()
    for identity in (
        "research-view",
        "history-view",
        "settings-view",
        "topic-question",
        "link-asset",
        "turn-select",
        "update-report",
        "stop-run",
        "resume-run",
        "copy-report",
        "save-report",
    ):
        assert f'id="{identity}"' in html
    for retired in (
        "openTrace",
        "loadOutcomeStats",
        "/runs/replay",
        "/watchlist",
        "/config/calibration",
        "collab-agents",
        "planner-mode",
    ):
        assert retired not in html + js
    assert "request_id" in js and "events?after=" in js
    assert "crypto.randomUUID()" in js
