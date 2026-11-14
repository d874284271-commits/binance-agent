"""M7 安全加固测试:Fernet 加密密钥库 round-trip + 日志脱敏。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from binance_agent.infra import logging as ilog
from binance_agent.infra.secrets import SecretStore, SecretStoreError


def test_secret_store_roundtrip(tmp_path: Path) -> None:
    store = SecretStore(Fernet(Fernet.generate_key()), tmp_path / "secrets.enc")
    assert store.get("BINANCE_API_KEY") is None
    store.set("BINANCE_API_KEY", "topsecret123456")
    assert store.get("BINANCE_API_KEY") == "topsecret123456"
    # 密文落盘不含明文
    raw = (tmp_path / "secrets.enc").read_bytes()
    assert b"topsecret123456" not in raw


def test_secret_store_wrong_key_preserves_original(tmp_path: Path) -> None:
    path = tmp_path / "secrets.enc"
    SecretStore(Fernet(Fernet.generate_key()), path).set("A", "value")
    # 换 key 不能把不可读原件当作空库覆盖。
    other = SecretStore(Fernet(Fernet.generate_key()), path)
    original = path.read_bytes()
    with pytest.raises(SecretStoreError):
        other.get("A")
    with pytest.raises(SecretStoreError):
        other.set("B", "new")
    assert path.read_bytes() == original


def test_log_redaction_masks_secrets() -> None:
    ilog.register_secret("supersecretvalue123")
    filt = ilog.RedactingFilter()
    record = logging.LogRecord(
        "n",
        logging.INFO,
        "p",
        1,
        "key=supersecretvalue123 token=sk-ABCDEFGH12345678",
        (),
        None,
    )
    assert filt.filter(record) is True
    message = record.getMessage()
    assert "supersecretvalue123" not in message
    assert "sk-ABCDEFGH12345678" not in message
    assert "REDACTED" in message
