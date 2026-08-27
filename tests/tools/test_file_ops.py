from __future__ import annotations

import pytest

from vela.policy.path_guard import PathPolicyError
from vela.tools import file_ops as fops

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def test_resolve_path_always_enforces_workspace_guard(tmp_path):
    with pytest.raises(PathPolicyError):
        fops.resolve_path(str(tmp_path), "../outside.txt")

    absolute = tmp_path / "abs.txt"
    assert fops.resolve_path(str(tmp_path), str(absolute)) == absolute


def test_skip_file_ignores_skipped_dirs_oversized_and_missing_files(tmp_path):
    inside_git = tmp_path / ".git" / "config"
    inside_git.parent.mkdir()
    inside_git.write_text("data", encoding="utf-8")
    small = tmp_path / "small.txt"
    small.write_text("data", encoding="utf-8")
    big = tmp_path / "big.txt"
    big.write_text("x" * (fops.MAX_FILE_SIZE + 1), encoding="utf-8")

    assert fops.skip_file(inside_git)
    assert fops.skip_file(big)
    assert fops.skip_file(tmp_path / "missing.txt")
    assert not fops.skip_file(small)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def test_read_file_numbers_lines_and_applies_offset_and_limit(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("a\nb\nc\nd\n", encoding="utf-8")

    full = fops.read_file(str(tmp_path), "notes.txt")
    window = fops.read_file(str(tmp_path), "notes.txt", offset=2, limit=2)

    assert not full.is_error
    assert full.content == "1: a\n2: b\n3: c\n4: d"
    assert full.display_summary == "Read notes.txt"
    assert window.content == "2: b\n3: c"


def test_read_file_clamps_non_positive_offset(tmp_path):
    (tmp_path / "notes.txt").write_text("a\nb\n", encoding="utf-8")

    result = fops.read_file(str(tmp_path), "notes.txt", offset=0)

    assert result.content == "1: a\n2: b"


def test_read_file_reports_missing_file_and_read_errors(tmp_path, monkeypatch):
    missing = fops.read_file(str(tmp_path), "nope.txt")
    assert missing.is_error
    assert "Not a file" in missing.content

    target = tmp_path / "broken.txt"
    target.write_text("data", encoding="utf-8")

    def fail_read(*args, **kwargs):
        raise OSError("device error")

    monkeypatch.setattr("pathlib.Path.read_text", fail_read)
    failed = fops.read_file(str(tmp_path), "broken.txt")

    assert failed.is_error
    assert "Failed to read" in failed.content


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def test_write_file_creates_parents_and_appends(tmp_path):
    created = fops.write_file(str(tmp_path), "nested/dir/out.txt", "first\n")
    appended = fops.write_file(str(tmp_path), "nested/dir/out.txt", "second\n", append=True)

    assert not created.is_error
    assert created.display_summary == "Wrote nested/dir/out.txt"
    assert not appended.is_error
    written = tmp_path / "nested" / "dir" / "out.txt"
    assert written.read_text(encoding="utf-8") == "first\nsecond\n"


def test_write_file_rejects_oversized_content(tmp_path):
    result = fops.write_file(str(tmp_path), "big.txt", "x" * (5 * 1024 * 1024 + 1))

    assert result.is_error
    assert "exceeds 5 MB" in result.content
    assert not (tmp_path / "big.txt").exists()


def test_write_file_reports_os_errors(tmp_path, monkeypatch):
    def fail_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.open", fail_open)
    result = fops.write_file(str(tmp_path), "out.txt", "content")

    assert result.is_error
    assert "Failed to write" in result.content


def test_write_file_appends_python_diagnostics(tmp_path):
    broken = fops.write_file(str(tmp_path), "broken.py", "def f(:\n")
    clean = fops.write_file(str(tmp_path), "clean.py", "def f():\n    return 1\n")
    skipped = fops.write_file(str(tmp_path), "also_broken.py", "def f(:\n", run_diagnostics=False)

    assert "Diagnostics:" in broken.content
    assert "Diagnostics:" not in clean.content
    assert "Diagnostics:" not in skipped.content


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------


def test_edit_file_replaces_first_occurrence_only(tmp_path):
    target = tmp_path / "code.txt"
    target.write_text("alpha\nalpha\n", encoding="utf-8")

    result = fops.edit_file(str(tmp_path), "code.txt", "alpha", "beta")

    assert not result.is_error
    assert target.read_text(encoding="utf-8") == "beta\nalpha\n"
    assert result.display_summary == "Edited code.txt"
    assert "-alpha" in result.content
    assert "+beta" in result.content


def test_edit_file_dry_run_keeps_file_unchanged(tmp_path):
    target = tmp_path / "code.txt"
    target.write_text("alpha\n", encoding="utf-8")

    result = fops.edit_file(str(tmp_path), "code.txt", "alpha", "beta", dry_run=True)

    assert not result.is_error
    assert result.content.startswith("[DRY RUN] Would edit code.txt")
    assert result.display_summary == "Dry-run edit code.txt"
    assert target.read_text(encoding="utf-8") == "alpha\n"


def test_edit_file_requires_existing_old_text(tmp_path):
    target = tmp_path / "code.txt"
    target.write_text("alpha\n", encoding="utf-8")

    missing_file = fops.edit_file(str(tmp_path), "nope.txt", "a", "b")
    missing_text = fops.edit_file(str(tmp_path), "code.txt", "gamma", "beta")

    assert missing_file.is_error
    assert "Not a file" in missing_file.content
    assert missing_text.is_error
    assert "`old_text` not found" in missing_text.content


def test_edit_file_reports_noop_when_texts_are_identical(tmp_path):
    target = tmp_path / "code.txt"
    target.write_text("alpha\n", encoding="utf-8")

    result = fops.edit_file(str(tmp_path), "code.txt", "alpha", "alpha")

    assert not result.is_error
    assert "no changes made" in result.content


def test_edit_file_reports_read_and_write_errors(tmp_path, monkeypatch):
    target = tmp_path / "code.txt"
    target.write_text("alpha\n", encoding="utf-8")

    real_open = type(target).open

    def guarded_open(self, mode="r", *args, **kwargs):
        if mode.startswith("w"):
            raise OSError("read-only file system")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr("pathlib.Path.open", guarded_open)
    write_failed = fops.edit_file(str(tmp_path), "code.txt", "alpha", "beta")

    assert write_failed.is_error
    assert "Failed to write" in write_failed.content

    def fail_read(*args, **kwargs):
        raise OSError("device error")

    monkeypatch.setattr("pathlib.Path.read_text", fail_read)
    read_failed = fops.edit_file(str(tmp_path), "code.txt", "alpha", "beta")

    assert read_failed.is_error
    assert "Failed to read" in read_failed.content


def test_edit_file_appends_python_diagnostics(tmp_path):
    target = tmp_path / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")

    result = fops.edit_file(str(tmp_path), "module.py", "value = 1", "value = (1")

    assert not result.is_error
    assert "Diagnostics:" in result.content


def test_build_diff_summary_switches_to_counts_for_large_edits():
    small = fops._build_diff_summary("a\nb", "c")
    large = fops._build_diff_summary("\n".join(str(i) for i in range(6)), "x")

    assert small == "-a\n-b\n+c"
    assert large == "--- removed 6 line(s)\n+++ added 1 line(s)"


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def test_list_directory_marks_directories_first(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "empty").mkdir()

    listed = fops.list_directory(str(tmp_path), ".")
    empty = fops.list_directory(str(tmp_path), "empty")
    not_a_dir = fops.list_directory(str(tmp_path), "a.txt")

    assert listed.content == "empty/\nsub/\na.txt\nb.txt"
    assert empty.content == "(empty directory)"
    assert not_a_dir.is_error
    assert "Not a directory" in not_a_dir.content


def test_list_directory_reports_iteration_errors(tmp_path, monkeypatch):
    def fail_iterdir(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("pathlib.Path.iterdir", fail_iterdir)
    result = fops.list_directory(str(tmp_path), ".")

    assert result.is_error
    assert "Failed to list" in result.content


# ---------------------------------------------------------------------------
# Grep
# ---------------------------------------------------------------------------


def test_grep_searches_recursively_and_reports_line_numbers(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("import os\nvalue = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("value = 2\n", encoding="utf-8")

    result = fops.grep(str(tmp_path), r"value = \d")

    assert not result.is_error
    assert "pkg/a.py:2: value = 1" in result.content
    assert "pkg/b.py:1: value = 2" in result.content


def test_grep_supports_plain_substring_single_file_and_limit(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("a.b\nxaxbx\na.b\n", encoding="utf-8")

    literal = fops.grep(str(tmp_path), "a.b", path="a.txt", use_regex=False)
    limited = fops.grep(str(tmp_path), "a.b", path="a.txt", use_regex=False, limit=1)
    empty = fops.grep(str(tmp_path), "zzz", path="a.txt", use_regex=False)

    assert literal.content == "a.txt:1: a.b\na.txt:3: a.b"
    assert limited.content == "a.txt:1: a.b"
    assert empty.content == "(no matches)"


def test_grep_rejects_invalid_regex_and_skips_unreadable_files(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("value\n", encoding="utf-8")

    invalid = fops.grep(str(tmp_path), "value(")
    assert invalid.is_error
    assert "invalid regex" in invalid.content

    def fail_read(*args, **kwargs):
        raise OSError("device error")

    monkeypatch.setattr("pathlib.Path.read_text", fail_read)
    skipped = fops.grep(str(tmp_path), "value")

    assert not skipped.is_error
    assert skipped.content == "(no matches)\n\n(skipped 1 unreadable file: a.txt: device error)"


def test_grep_skips_files_inside_skipped_directories(tmp_path):
    cached = tmp_path / "__pycache__"
    cached.mkdir()
    (cached / "stale.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "fresh.txt").write_text("needle\n", encoding="utf-8")

    result = fops.grep(str(tmp_path), "needle")

    assert result.content == "fresh.txt:1: needle"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_relative_to_falls_back_to_absolute_path(tmp_path):
    outside = tmp_path.parent / "outside.txt"

    assert fops._relative_to(tmp_path / "inside.txt", str(tmp_path)).as_posix() == "inside.txt"
    assert fops._relative_to(outside, str(tmp_path)) == outside
