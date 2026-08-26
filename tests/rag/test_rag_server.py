from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from vela_rag import server as rag_server
from vela_rag.index import CodeIndex


def _project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "service.py").write_text(
        "def resume_session(session_id):\n    return store.load(session_id)\n",
        encoding="utf-8",
    )
    return root


def _call(server, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = asyncio.run(server.call_tool(name, arguments))
    if isinstance(result, dict):
        return result
    blocks, structured = result
    if isinstance(structured, dict):
        return structured
    return json.loads(blocks[0].text)


def test_create_server_exposes_read_only_search_and_status_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    server = rag_server.create_server(_project(tmp_path))

    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}

    assert set(tools) == {"search_code", "rag_status"}
    assert tools["search_code"].annotations.readOnlyHint is True
    assert tools["rag_status"].annotations.readOnlyHint is True


def test_search_code_indexes_the_project_and_returns_line_references(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = _project(tmp_path)
    server = rag_server.create_server(root)

    payload = _call(server, "search_code", {"query": "resume_session"})

    assert payload["query"] == "resume_session"
    assert payload["results"]
    assert payload["results"][0]["path"] == "service.py"
    assert payload["results"][0]["start_line"] >= 1
    assert "warning" not in payload


def test_search_code_clamps_the_limit_to_the_supported_range(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = _project(tmp_path)
    limits: list[int] = []
    real_search = CodeIndex.search

    def recording_search(self, query, *, limit=8):
        limits.append(limit)
        return real_search(self, query, limit=limit)

    monkeypatch.setattr(CodeIndex, "search", recording_search)
    server = rag_server.create_server(root)

    _call(server, "search_code", {"query": "resume_session", "limit": 999})
    _call(server, "search_code", {"query": "resume_session", "limit": 0})

    assert limits == [20, 1]


@pytest.mark.parametrize("stage", ["rebuild", "search"])
def test_search_code_surfaces_index_warnings_from_either_stage(tmp_path, monkeypatch, stage):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = _project(tmp_path)
    real_rebuild = CodeIndex.rebuild
    real_search = CodeIndex.search

    def warning_rebuild(self):
        stats = real_rebuild(self)
        if stage == "rebuild":
            self.last_warning = "embedding unavailable during rebuild"
        return stats

    def warning_search(self, query, *, limit=8):
        hits = real_search(self, query, limit=limit)
        if stage == "search":
            self.last_warning = "embedding unavailable during search"
        else:
            self.last_warning = None
        return hits

    monkeypatch.setattr(CodeIndex, "rebuild", warning_rebuild)
    monkeypatch.setattr(CodeIndex, "search", warning_search)
    server = rag_server.create_server(root)

    payload = _call(server, "search_code", {"query": "resume_session"})

    assert payload["warning"] == f"embedding unavailable during {stage}"


def test_rag_status_reports_index_location_and_retrieval_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = _project(tmp_path)
    server = rag_server.create_server(root)

    _call(server, "search_code", {"query": "resume_session"})
    payload = _call(server, "rag_status", {})

    assert payload["root"] == str(root.resolve())
    assert payload["database"].endswith(".sqlite")
    assert payload["files"] == 1
    assert payload["chunks"] >= 1
    assert payload["retrieval_mode"] == "lexical"


def test_create_server_enables_hybrid_retrieval_from_embedding_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VELA_RAG_EMBEDDING_API_KEY", "rag-secret")
    monkeypatch.setenv("VELA_RAG_EMBEDDING_MODEL", "embedding-model")
    root = _project(tmp_path)

    server = rag_server.create_server(root)
    payload = _call(server, "rag_status", {})

    assert payload["retrieval_mode"] == "hybrid"
    assert server.name == "Vela Code RAG"


def test_main_runs_the_server_over_stdio_for_the_requested_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = _project(tmp_path)
    transports: list[str] = []

    monkeypatch.setattr("sys.argv", ["vela-rag", "--root", str(root)])
    monkeypatch.setattr(
        rag_server.FastMCP,
        "run",
        lambda self, transport="stdio": transports.append(transport),
    )

    rag_server.main()

    assert transports == ["stdio"]


def test_main_requires_a_root_argument(monkeypatch):
    monkeypatch.setattr("sys.argv", ["vela-rag"])

    with pytest.raises(SystemExit):
        rag_server.main()
