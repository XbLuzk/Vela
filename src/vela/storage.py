"""Shared helpers for Vela's private user-level state on disk.

Vela keeps sessions, traces, journals, caches and trust decisions under
``~/.vela`` with owner-only permissions. These helpers centralize the
directory layout, the permission bits and the SQLite connection settings that
every store relies on.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

VELA_DIR_NAME = ".vela"
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
SQLITE_PRAGMAS = ("journal_mode=WAL", "busy_timeout=5000")
DEFAULT_LOCK_TIMEOUT = 5.0

__all__ = [
    "DEFAULT_LOCK_TIMEOUT",
    "PRIVATE_DIR_MODE",
    "PRIVATE_FILE_MODE",
    "SQLITE_PRAGMAS",
    "VELA_DIR_NAME",
    "apply_sqlite_pragmas",
    "ensure_private_dir",
    "ensure_private_file",
    "exclusive_lock",
    "set_private_mode",
    "user_state_path",
    "vela_dir",
    "vela_home",
    "write_private_text",
]


def vela_home() -> Path:
    """Return the user-level Vela directory (``~/.vela``)."""
    return Path.home() / VELA_DIR_NAME


def vela_dir(root: str | Path) -> Path:
    """Return the project-level (or arbitrary root) ``.vela`` directory."""
    return Path(root) / VELA_DIR_NAME


def user_state_path(*parts: str) -> Path:
    """Return a path inside ``~/.vela`` without creating anything."""
    return vela_home().joinpath(*parts)


def set_private_mode(path: str | Path, mode: int, *, verify: bool = False) -> None:
    """Apply ``mode`` to ``path``, optionally verifying that POSIX accepted it."""
    target = Path(path)
    os.chmod(target, mode)
    if verify and os.name == "posix" and target.stat().st_mode & 0o777 != mode:
        raise PermissionError(f"Could not secure private path: {target}")


def ensure_private_dir(path: str | Path, *, verify: bool = False) -> Path:
    """Create ``path`` and its parents as an owner-only directory."""
    directory = Path(path).expanduser()
    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    directory.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)

    targets = [directory, *missing]
    if VELA_DIR_NAME in directory.parts:
        current = directory
        while current.name != VELA_DIR_NAME and current.parent != current:
            current = current.parent
            targets.append(current)
    for target in dict.fromkeys(reversed(targets)):
        set_private_mode(target, PRIVATE_DIR_MODE, verify=verify)
    return directory


def ensure_private_file(path: str | Path, *, verify: bool = False) -> Path:
    """Create ``path`` and its private parent directory as an owner-only file."""
    target = Path(path).expanduser()
    ensure_private_dir(target.parent, verify=verify)
    descriptor = os.open(target, os.O_CREAT | os.O_RDWR, PRIVATE_FILE_MODE)
    try:
        set_private_mode(target, PRIVATE_FILE_MODE, verify=verify)
    finally:
        os.close(descriptor)
    return target


def write_private_text(path: str | Path, text: str) -> None:
    """Replace ``path`` with ``text`` atomically, keeping owner-only permissions."""
    target = Path(path).expanduser()
    ensure_private_dir(target.parent)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            PRIVATE_FILE_MODE,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(text)
        temporary.replace(target)
        set_private_mode(target, PRIVATE_FILE_MODE)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(
    path: str | Path,
    *,
    busy_message: str,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> Iterator[None]:
    """Hold ``<path>.lock`` exclusively, reporting contention as ``OSError``."""
    lock_path = Path(f"{path}.lock")
    ensure_private_file(lock_path)
    try:
        with FileLock(lock_path, timeout=timeout):
            set_private_mode(lock_path, PRIVATE_FILE_MODE)
            yield
    except Timeout as exc:
        raise OSError(busy_message) from exc


def apply_sqlite_pragmas(connection: sqlite3.Connection) -> None:
    """Apply Vela's shared durability and contention PRAGMAs."""
    for pragma in SQLITE_PRAGMAS:
        connection.execute(f"PRAGMA {pragma}")
