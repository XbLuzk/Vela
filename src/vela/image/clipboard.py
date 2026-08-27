from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from vela.storage import (
    PRIVATE_FILE_MODE,
    ensure_private_dir,
    set_private_mode,
    user_state_path,
)


@dataclass(frozen=True, slots=True)
class ClipboardImageResult:
    ok: bool
    path: Path | None = None
    error: str = ""

    @classmethod
    def success(cls, path: Path) -> ClipboardImageResult:
        return cls(True, path=path)

    @classmethod
    def failure(cls, error: str) -> ClipboardImageResult:
        return cls(False, error=error)


CommandRunner = Callable[..., subprocess.CompletedProcess]


def grab_clipboard_image(
    cache_dir: str | Path | None = None,
    *,
    platform: str | None = None,
    runner: CommandRunner = subprocess.run,
) -> ClipboardImageResult:
    """Save the current macOS clipboard image under ``~/.vela/cache``."""

    current_platform = platform or sys.platform
    if current_platform != "darwin":
        return ClipboardImageResult.failure("当前仅支持在 macOS 读取剪贴板图片")

    target_dir = Path(cache_dir or user_state_path("cache")).expanduser()
    png_path: Path | None = None
    tiff_path: Path | None = None
    keep_png = False
    try:
        ensure_private_dir(target_dir)
        stamp = time.time_ns()
        png_path = target_dir / f"clip-{stamp}.png"
        png_result = _run_osascript(_MAC_CLIPBOARD_PNG_SCRIPT, png_path, runner)
        if _is_nonempty_file(png_path) and png_result.returncode == 0:
            set_private_mode(png_path, PRIVATE_FILE_MODE)
            keep_png = True
            return ClipboardImageResult.success(png_path)
        png_path.unlink(missing_ok=True)

        tiff_path = target_dir / f"clip-{stamp}.tiff"
        tiff_result = _run_osascript(_MAC_CLIPBOARD_TIFF_SCRIPT, tiff_path, runner)
        conversion: subprocess.CompletedProcess | None = None
        if tiff_result.returncode == 0 and _is_nonempty_file(tiff_path):
            conversion = runner(
                [
                    "/usr/bin/sips",
                    "-s",
                    "format",
                    "png",
                    str(tiff_path),
                    "--out",
                    str(png_path),
                ],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if conversion.returncode == 0 and _is_nonempty_file(png_path):
                set_private_mode(png_path, PRIVATE_FILE_MODE)
                keep_png = True
                return ClipboardImageResult.success(png_path)
        detail = (
            (conversion.stderr if conversion is not None else "")
            or tiff_result.stderr
            or png_result.stderr
            or ""
        ).strip()
        return ClipboardImageResult.failure(detail or "剪贴板里没有图片，请先截图后再按 Ctrl+V")
    except subprocess.TimeoutExpired:
        return ClipboardImageResult.failure("读取剪贴板图片超时")
    except Exception as exc:  # noqa: BLE001 - clipboard failure must not break input
        return ClipboardImageResult.failure(f"读取剪贴板图片失败: {exc}")
    finally:
        if tiff_path is not None:
            with suppress(OSError):
                tiff_path.unlink(missing_ok=True)
        if png_path is not None and not keep_png:
            with suppress(OSError):
                png_path.unlink(missing_ok=True)


def _run_osascript(
    script: str,
    output_path: Path,
    runner: CommandRunner,
) -> subprocess.CompletedProcess:
    return runner(
        ["/usr/bin/osascript", "-", str(output_path)],
        input=script,
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


_MAC_CLIPBOARD_PNG_SCRIPT = """
on run argv
    set outputPath to item 1 of argv
    try
        set pngData to (the clipboard as «class PNGf»)
    on error
        error "剪贴板里没有 PNG 数据"
    end try
    set fh to open for access (POSIX file outputPath as string) with write permission
    try
        set eof of fh to 0
        write pngData to fh
        close access fh
    on error errMsg
        try
            close access fh
        end try
        error errMsg
    end try
end run
"""


_MAC_CLIPBOARD_TIFF_SCRIPT = """
on run argv
    set outputPath to item 1 of argv
    try
        set tiffData to (the clipboard as «class TIFF»)
    on error
        error "剪贴板里没有 TIFF 数据"
    end try
    set fh to open for access (POSIX file outputPath as string) with write permission
    try
        set eof of fh to 0
        write tiffData to fh
        close access fh
    on error errMsg
        try
            close access fh
        end try
        error errMsg
    end try
end run
"""
