"""全局快捷键 + 剪贴板同步:复制交易对 → 按热键 → 触发分析。

pynput / pyperclip 懒导入(仅 start/触发时),保持模块在无桌面依赖环境可导入与单测。
热键回调拿到清洗后的交易对,由桌面壳注入悬浮窗执行分析。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_CLEAN_RE = re.compile(r"[^A-Z0-9]")
_QUOTES = ("USDT", "USDC", "FDUSD", "BUSD", "TUSD", "USD", "DAI")


def sanitize_symbol(text: str | None) -> str | None:
    """剪贴板文本 → 规范交易对。无效返回 None。

    'eth' → 'ETHUSDT';'BTC/USDT' → 'BTCUSDT';'ETHUSDT Perp' → 'ETHUSDT'。
    """
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    token = stripped.split()[0]
    symbol = _CLEAN_RE.sub("", token.upper())
    if len(symbol) < 2 or len(symbol) > 20:
        return None
    if not symbol.endswith(_QUOTES):
        symbol += "USDT"
    return symbol


class HotkeyListener:
    """监听全局热键,触发时读剪贴板 → 清洗 → 回调交易对。"""

    def __init__(self, combo: str, on_trigger: Callable[[str], None]) -> None:
        self._combo = combo
        self._on_trigger = on_trigger
        self._listener: Any = None

    def _activate(self) -> None:
        try:
            import pyperclip

            clip = pyperclip.paste()
        except Exception as exc:
            logger.warning("读剪贴板失败: %s", exc)
            return
        symbol = sanitize_symbol(clip)
        if symbol:
            logger.info("热键触发分析: %s", symbol)
            self._on_trigger(symbol)
        else:
            logger.info("剪贴板内容非有效交易对,忽略")

    def start(self) -> None:
        from pynput import keyboard

        self._listener = keyboard.GlobalHotKeys({self._combo: self._activate})
        self._listener.start()
        logger.info("全局热键已注册: %s", self._combo)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
