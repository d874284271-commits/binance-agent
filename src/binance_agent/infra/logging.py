"""脱敏日志:防止币安/LLM/数据源密钥泄漏进日志。

策略:运行时把已解析的密钥值注册进集合,日志输出前统一替换为掩码;
另对常见 OpenAI 风格 ``sk-`` 前缀串做正则兜底。
"""

from __future__ import annotations

import logging
import re
import sys

_REDACTED = "***REDACTED***"
_registered_secrets: set[str] = set()
# 兜底:OpenAI 风格密钥(sk-...);已注册的真实密钥优先精确替换
_KEYLIKE_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


def register_secret(value: str | None) -> None:
    """注册一个需要在日志中脱敏的密钥值(过短的忽略以免误伤)。"""
    if value and len(value) >= 6:
        _registered_secrets.add(value)


def redact_text(text: str) -> str:
    """对任意字符串做密钥脱敏:已注册密钥精确替换 + sk- 前缀正则兜底。

    供日志过滤器以及落盘/推流路径(RunStore JSONL、SSE)复用,
    确保密钥不经任何输出通道泄漏(不止 logging)。
    """
    for secret in _registered_secrets:
        text = text.replace(secret, _REDACTED)
    return _KEYLIKE_RE.sub(_REDACTED, text)


class RedactingFilter(logging.Filter):
    """日志过滤器:对最终消息文本做脱敏替换。"""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = redact_text(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(level: str = "INFO") -> None:
    """配置根日志:统一格式 + 脱敏过滤器;冻结包(无控制台)追加滚动文件日志。"""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    handler: logging.Handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(RedactingFilter())
    handlers: list[logging.Handler] = [handler]

    if getattr(sys, "frozen", False):  # 打包后窗口模式无 stderr,日志落 ~/.binance_agent/logs
        from logging.handlers import RotatingFileHandler

        from .paths import data_dir

        log_path = data_dir() / "logs" / "agent.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RedactingFilter())
        handlers.append(file_handler)

    root = logging.getLogger()
    root.handlers[:] = handlers
    root.setLevel(level.upper())
