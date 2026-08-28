from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class DirectoryPickerUnavailable(RuntimeError):
    pass


def pick_directory(current: str | Path) -> str | None:
    """Open the host operating system's folder chooser without invoking a shell."""
    cwd = Path(current).expanduser().resolve()
    if sys.platform == "darwin":
        return _pick_macos(cwd)
    if sys.platform == "win32":
        return _pick_windows(cwd)
    return _pick_linux(cwd)


def _pick_macos(current: Path) -> str | None:
    script = """
on run argv
  set currentFolder to POSIX file (item 1 of argv)
  set promptText to "选择 Vela 项目目录"
  set selectedFolder to choose folder with prompt promptText default location currentFolder
  return POSIX path of selectedFolder
end run
"""
    return _run_picker(["osascript", "-e", script, str(current)], cancelled_codes={1})


def _pick_windows(current: Path) -> str | None:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        raise DirectoryPickerUnavailable("系统目录选择器不可用；未找到 PowerShell。")
    script = """
param([string]$InitialPath)
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '选择 Vela 项目目录'
$dialog.SelectedPath = $InitialPath
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
  [Console]::Out.Write($dialog.SelectedPath)
}
"""
    return _run_picker(
        [executable, "-NoProfile", "-STA", "-Command", script, str(current)],
        cancelled_codes=set(),
    )


def _pick_linux(current: Path) -> str | None:
    zenity = shutil.which("zenity")
    if zenity is not None:
        return _run_picker(
            [
                zenity,
                "--file-selection",
                "--directory",
                "--title=选择 Vela 项目目录",
                f"--filename={current}/",
            ],
            cancelled_codes={1},
        )
    kdialog = shutil.which("kdialog")
    if kdialog is not None:
        return _run_picker(
            [kdialog, "--getexistingdirectory", str(current), "--title", "选择 Vela 项目目录"],
            cancelled_codes={1},
        )
    raise DirectoryPickerUnavailable("系统目录选择器不可用；请安装 zenity 或 kdialog。")


def _run_picker(command: list[str], *, cancelled_codes: set[int]) -> str | None:
    result = subprocess.run(command, capture_output=True, check=False, text=True)  # noqa: S603
    if result.returncode in cancelled_codes:
        return None
    if result.returncode != 0:
        detail = result.stderr.strip() or "系统目录选择器启动失败"
        raise DirectoryPickerUnavailable(detail)
    selected = result.stdout.strip()
    return selected or None
