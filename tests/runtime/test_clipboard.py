from __future__ import annotations

import subprocess

from vela.image import grab_clipboard_image


def test_macos_clipboard_png_is_saved_in_private_cache(tmp_path):
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "/usr/bin/osascript":
            output_path = command[-1]
            with open(output_path, "wb") as image_file:
                image_file.write(b"png-data")
        return subprocess.CompletedProcess(command, 0, "", "")

    result = grab_clipboard_image(tmp_path / "cache", platform="darwin", runner=runner)

    assert result.ok
    assert result.path is not None and result.path.read_bytes() == b"png-data"
    assert result.path.stat().st_mode & 0o777 == 0o600
    assert calls[0][0][0] == "/usr/bin/osascript"


def test_clipboard_reports_no_image_without_leaving_cache_files(tmp_path):
    def runner(command, **kwargs):  # noqa: ARG001
        return subprocess.CompletedProcess(command, 1, "", "no image")

    result = grab_clipboard_image(tmp_path / "cache", platform="darwin", runner=runner)

    assert not result.ok
    assert "no image" in result.error
    assert list((tmp_path / "cache").iterdir()) == []


def test_macos_clipboard_falls_back_from_tiff_and_removes_intermediate(tmp_path):
    def runner(command, **kwargs):
        if command[0] == "/usr/bin/osascript":
            if "PNGf" in kwargs["input"]:
                return subprocess.CompletedProcess(command, 1, "", "no png")
            with open(command[-1], "wb") as image_file:
                image_file.write(b"tiff-data")
            return subprocess.CompletedProcess(command, 0, "", "")
        output_path = command[-1]
        with open(output_path, "wb") as image_file:
            image_file.write(b"converted-png")
        return subprocess.CompletedProcess(command, 0, "", "")

    result = grab_clipboard_image(tmp_path / "cache", platform="darwin", runner=runner)

    assert result.ok
    assert result.path is not None and result.path.read_bytes() == b"converted-png"
    assert list((tmp_path / "cache").glob("*.tiff")) == []


def test_clipboard_timeout_cleans_partial_files(tmp_path):
    def runner(command, **kwargs):  # noqa: ARG001
        with open(command[-1], "wb") as image_file:
            image_file.write(b"partial")
        raise subprocess.TimeoutExpired(command, 8)

    result = grab_clipboard_image(tmp_path / "cache", platform="darwin", runner=runner)

    assert not result.ok
    assert "超时" in result.error
    assert list((tmp_path / "cache").iterdir()) == []


def test_failed_tiff_conversion_reports_final_error_and_removes_partial_png(tmp_path):
    def runner(command, **kwargs):
        if command[0] == "/usr/bin/osascript":
            if "PNGf" in kwargs["input"]:
                return subprocess.CompletedProcess(command, 1, "", "no png")
            with open(command[-1], "wb") as image_file:
                image_file.write(b"tiff-data")
            return subprocess.CompletedProcess(command, 0, "", "")
        with open(command[-1], "wb") as image_file:
            image_file.write(b"partial-png")
        return subprocess.CompletedProcess(command, 1, "", "conversion failed")

    result = grab_clipboard_image(tmp_path / "cache", platform="darwin", runner=runner)

    assert not result.ok
    assert "conversion failed" in result.error
    assert list((tmp_path / "cache").iterdir()) == []


def test_clipboard_is_explicitly_unsupported_off_macos(tmp_path):
    result = grab_clipboard_image(tmp_path / "cache", platform="linux")

    assert not result.ok
    assert "macOS" in result.error
