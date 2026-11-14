"""密钥解析与加密存储。

解析顺序:环境变量(含 .env)→ 本地 Fernet 加密库 → keyring 直存。
币安等敏感 Key 可经 ``store_secret`` 加密落盘(密文存盘,Fernet key 存系统钥匙串),
**严禁**明文落盘。取到的值统一注册进日志脱敏集合。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .file_io import atomic_write, file_lock
from .logging import register_secret

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "binance-agent"
_FERNET_KEY_NAME = "fernet-key"
_STORE_PATH = Path.home() / ".binance_agent" / "secrets.enc"


class SecretStoreError(ValueError):
    """已有密文不可读取;禁止把解密失败当作空库覆盖。"""


class SecretStore:
    """Fernet 加密的本地密钥库:密文落盘,Fernet key 由系统钥匙串保管。"""

    def __init__(self, fernet: Any, path: Path) -> None:
        self._fernet = fernet
        self._path = path

    def _load(self) -> dict[str, str]:
        """解密并验证字符串映射;文件缺失返回空库,损坏或解密失败保留原件并抛错。"""
        if not self._path.exists():
            return {}
        token = self._path.read_bytes()
        try:
            decrypted = self._fernet.decrypt(token)
            data = json.loads(decrypted.decode())
            if not isinstance(data, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in data.items()
            ):
                raise ValueError("invalid secret mapping")
            return data
        except Exception as exc:
            raise SecretStoreError(f"无法解密本地密钥库 {self._path}; 原文件保持不变") from exc

    def get(self, name: str) -> str | None:
        with file_lock(self._path):
            return self._load().get(name)

    def set(self, name: str, value: str) -> None:
        """持跨进程锁完成读改写,再原子替换密文;POSIX 文件权限为 0600。"""
        with file_lock(self._path):
            data = self._load()
            data[name] = value
            atomic_write(self._path, self._fernet.encrypt(json.dumps(data).encode()), private=True)


def _default_store() -> SecretStore | None:
    try:
        import keyring
        from cryptography.fernet import Fernet

        # Key creation must share the encrypted-store lock: independent processes
        # must not encrypt their first writes using competing freshly generated keys.
        with file_lock(_STORE_PATH):
            key = keyring.get_password(KEYRING_SERVICE, _FERNET_KEY_NAME)
            if not key:
                if _STORE_PATH.exists():
                    raise SecretStoreError(
                        f"本地密钥库 {_STORE_PATH} 已存在但缺少解密密钥; 保留原件"
                    )
                key = Fernet.generate_key().decode()
                keyring.set_password(KEYRING_SERVICE, _FERNET_KEY_NAME, key)
        return SecretStore(Fernet(key.encode()), _STORE_PATH)
    except SecretStoreError:
        raise
    except Exception as exc:
        logger.debug("加密密钥库不可用(keyring/cryptography): %s", exc)
        return None


def store_secret(name: str, value: str) -> bool:
    """加密保存并登记日志脱敏值;系统密钥存储不可用返回 False,已有坏库/写入失败向上传递。"""
    store = _default_store()
    if store is None:
        return False
    store.set(name, value)
    register_secret(value)
    return True


def resolve_secret(ref: str | None) -> str | None:
    """依次从环境变量、加密库、系统钥匙串解析引用,空值继续后备;缺失返回 None。

    成功值登记进日志脱敏集合;此函数可能访问系统钥匙串,调用者不得把返回值写入响应或日志。
    """
    if not ref:
        return None

    value = os.environ.get(ref)  # 1) 环境变量 / .env

    if not value:  # 2) 本地 Fernet 加密库(空 env 值不应遮蔽加密库,故用 not value)
        store = _default_store()
        if store is not None:
            value = store.get(ref)

    if not value:  # 3) keyring 直存
        try:
            import keyring

            value = keyring.get_password(KEYRING_SERVICE, ref)
        except Exception as exc:
            logger.debug("keyring 不可用,跳过 %s: %s", ref, exc)
            value = None

    if value:
        register_secret(value)
        return value
    return None
