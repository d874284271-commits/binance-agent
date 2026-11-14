"""M6 热键/剪贴板:交易对清洗纯函数测试(pynput/pyperclip 懒导入,本测试无需它们)。"""

from __future__ import annotations

from binance_agent.app.hotkey import sanitize_symbol


def test_sanitize_symbol_variants() -> None:
    assert sanitize_symbol("ETHUSDT") == "ETHUSDT"
    assert sanitize_symbol("eth") == "ETHUSDT"  # 补 USDT
    assert sanitize_symbol("BTC/USDT") == "BTCUSDT"  # 去分隔符
    assert sanitize_symbol("ETHUSDT Perp") == "ETHUSDT"  # 取首 token
    assert sanitize_symbol("  btcusdc ") == "BTCUSDC"  # 已带 quote 不再补
    assert sanitize_symbol("solusdt\n") == "SOLUSDT"


def test_sanitize_symbol_invalid() -> None:
    assert sanitize_symbol("") is None
    assert sanitize_symbol(None) is None
    assert sanitize_symbol("   ") is None
    assert sanitize_symbol("x") is None  # 过短
    assert sanitize_symbol("!!!") is None  # 清洗后为空
