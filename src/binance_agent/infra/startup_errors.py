"""Actionable startup diagnostics without destructive recovery or secret values."""

from __future__ import annotations

from pathlib import Path


class StartupError(RuntimeError):
    """A failed startup resource, safe to display in the desktop or CLI."""

    def __init__(self, resource: str, path: Path | str, category: str, recovery: str) -> None:
        self.resource = resource
        self.path = Path(path)
        self.category = category
        super().__init__(f"启动失败: {resource} | 路径: {self.path} | 类别: {category}。{recovery}")


def startup_error(
    resource: str,
    path: Path | str,
    exc: BaseException,
    *,
    sqlite: bool = False,
) -> StartupError:
    """Build a safe diagnostic; callers retain the original error with ``from exc``."""
    recovery = "请检查该路径、目录权限及可用空间;保留原文件,修复后重新启动。"
    if sqlite:
        recovery = (
            "未自动重建数据库。请停止应用,完整备份原库及同名 -wal/-shm 文件;"
            "检查权限和空间,使用备份副本诊断或从已验证备份恢复后重新启动。"
        )
    return StartupError(resource, path, type(exc).__name__, recovery)
