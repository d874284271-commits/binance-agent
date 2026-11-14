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


def test_unlocked_store_reuses_keychain_key_but_reads_fresh_ciphertext(tmp_path, monkeypatch):
    import keyring

    from binance_agent.infra import secrets

    key = Fernet.generate_key()
    path = tmp_path / "secrets.enc"
    external = SecretStore(Fernet(key), path)
    external.set("MODEL", "first-value")
    calls = []
    monkeypatch.setattr(secrets, "_STORE_PATH", path)
    monkeypatch.setattr(secrets, "_cached_store", None)
    monkeypatch.setattr(secrets, "_unavailable_path", None)
    monkeypatch.setattr(keyring, "get_password", lambda *args: calls.append(args) or key.decode())
    secrets.prepare_secret_store()
    assert secrets.resolve_secret("MODEL") == "first-value"
    external.set("MODEL", "updated-value")
    assert secrets.resolve_secret("MODEL") == "updated-value"
    assert len(calls) == 1


def test_denied_keychain_is_not_reprompted_until_explicit_save(tmp_path, monkeypatch):
    import keyring

    from binance_agent.infra import secrets

    key = Fernet.generate_key()
    path = tmp_path / "secrets.enc"
    SecretStore(Fernet(key), path).set("MODEL", "keep-original")
    original = path.read_bytes()
    calls = []

    def get_password(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("user denied system authorization")
        return key.decode()

    monkeypatch.setattr(secrets, "_STORE_PATH", path)
    monkeypatch.setattr(secrets, "_cached_store", None)
    monkeypatch.setattr(secrets, "_unavailable_path", None)
    monkeypatch.setattr(keyring, "get_password", get_password)
    secrets.prepare_secret_store()
    assert secrets.resolve_secret("MODEL") is None
    assert secrets.resolve_secret("SEARCH") is None
    assert len(calls) == 1 and path.read_bytes() == original
    assert secrets.store_secret("NEW", "saved-after-explicit-action")
    assert secrets.resolve_secret("MODEL") == "keep-original"
    assert len(calls) == 2


def test_fresh_install_does_not_create_keychain_item_on_read(tmp_path, monkeypatch):
    import keyring

    from binance_agent.infra import secrets

    calls = []
    monkeypatch.setattr(secrets, "_STORE_PATH", tmp_path / "missing.enc")
    monkeypatch.setattr(secrets, "_cached_store", None)
    monkeypatch.setattr(secrets, "_unavailable_path", None)
    monkeypatch.setattr(keyring, "get_password", lambda *args: calls.append(args))
    monkeypatch.setattr(keyring, "set_password", lambda *args: pytest.fail("created unused key"))
    secrets.prepare_secret_store()
    assert calls == []
    assert secrets.resolve_secret("UNCONFIGURED") is None
    assert calls == [(secrets.KEYRING_SERVICE, "UNCONFIGURED")]
