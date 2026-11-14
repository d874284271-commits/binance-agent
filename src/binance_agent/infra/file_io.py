"""本地存储共用的文件锁、原子替换和 JSONL 读写工具,不执行数据迁移。"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_locks_guard = threading.Lock()
_locks: dict[str, threading.RLock] = {}


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """用固定旁路锁文件串行化协作线程/进程,目标文件被原子替换后锁仍有效。

    进入时会创建父目录和 .lock 文件;退出或异常时释放锁,不删除锁文件。
    锁只约束使用同一协议的调用者,不阻止其他程序直接改写目标文件。
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _locks_guard:
        lock = _locks.setdefault(str(lock_path.resolve()), threading.RLock())
    with lock, lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_write(path: Path, content: bytes, *, private: bool = False) -> None:
    """先写同目录临时文件并 fsync,再原子替换目标;异常向上传递并清理临时文件。

    private 在 POSIX 上把临时文件权限设为 0600;此函数不自行加锁,
    需要跨进程读改写互斥的调用者应在外层持有 file_lock。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            if private and os.name != "nt":
                os.fchmod(handle.fileno(), 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def append_jsonl(path: Path, line: str) -> None:
    """加锁追加一条 UTF-8 JSON 文本;先补齐已有半行的换行,避免吞掉新记录。

    调用者负责序列化单行 JSON;写后 flush,但不承诺断电后的持久性。
    """
    with file_lock(path), path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
        handle.write(line.encode("utf-8") + b"\n")
        handle.flush()


def decode_jsonl(raw: bytes, path: Path, line_number: int) -> dict[str, Any] | None:
    """严格解码一行 UTF-8 对象;空行返回 None,坏行记录路径/行号后跳过。"""
    if not raw.strip():
        return None
    try:
        data = json.loads(raw.decode("utf-8", errors="strict"))
        if not isinstance(data, dict):
            raise ValueError("record is not an object")
        return data
    except (ValueError, UnicodeError):
        logger.warning("跳过损坏 JSONL 记录: %s:%d", path, line_number)
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """持锁按行读取有效对象,沿用坏行隔离规则;文件打开错误交由调用者处理。"""
    records = []
    with file_lock(path), path.open("rb") as handle:
        for number, raw in enumerate(handle, 1):
            record = decode_jsonl(raw, path, number)
            if record is not None:
                records.append(record)
    return records
