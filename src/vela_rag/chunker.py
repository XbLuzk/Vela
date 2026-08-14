"""Deterministic source-file discovery and line-based chunking."""

from __future__ import annotations

import ast
import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

from vela_rag.models import CodeChunk

_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".xml",
    ".yaml",
    ".yml",
}
_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".vela",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}
_SPECIAL_FILES = {"Dockerfile", "Makefile"}
_MAX_FILE_BYTES = 1024 * 1024


def discover_source_files(root: Path) -> Iterator[Path]:
    """Yield supported text files below root in stable order."""
    resolved_root = root.resolve()
    for directory, directory_names, file_names in os.walk(resolved_root, followlinks=False):
        directory_path = Path(directory)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in _EXCLUDED_DIRS and not (directory_path / name).is_symlink()
        )
        for name in sorted(file_names):
            path = directory_path / name
            if path.is_symlink():
                continue
            if path.name not in _SPECIAL_FILES and path.suffix.lower() not in _EXTENSIONS:
                continue
            try:
                resolved = path.resolve()
                if not resolved.is_relative_to(resolved_root):
                    continue
                if name.endswith((".min.js", ".lock")) or resolved.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield resolved


def chunk_file(path: Path, root: Path, *, lines_per_chunk: int = 80) -> list[CodeChunk]:
    """Split one UTF-8 source file and attach the nearest Python symbol."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    lines = text.splitlines()
    if not lines:
        return []
    relative = path.relative_to(root).as_posix()
    symbols = _python_symbols(text) if path.suffix == ".py" else []
    chunks: list[CodeChunk] = []
    for offset in range(0, len(lines), lines_per_chunk):
        start = offset + 1
        end = min(len(lines), offset + lines_per_chunk)
        content = "\n".join(lines[offset:end]).strip()
        if not content:
            continue
        digest = hashlib.sha256(f"{relative}:{start}:{content}".encode()).hexdigest()[:20]
        chunks.append(
            CodeChunk(
                chunk_id=digest,
                path=relative,
                start_line=start,
                end_line=end,
                symbol=_nearest_symbol(symbols, start, end),
                content=content,
            )
        )
    return chunks


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _python_symbols(text: str) -> list[tuple[int, int, str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    symbols: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append((node.lineno, node.end_lineno or node.lineno, node.name))
    return symbols


def _nearest_symbol(symbols: list[tuple[int, int, str]], start: int, end: int) -> str:
    matches = [item for item in symbols if item[0] <= end and item[1] >= start]
    if not matches:
        return ""
    return min(matches, key=lambda item: item[1] - item[0])[2]
