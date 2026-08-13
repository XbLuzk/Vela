from __future__ import annotations

from vela_rag.index import CodeIndex


class FakeEmbedder:
    @property
    def identity(self) -> str:
        return "fake-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                float(any(word in text.lower() for word in ("payment", "charge", "card"))),
                float("session" in text.lower()),
            ]
            for text in texts
        ]


def test_code_index_returns_file_and_line_references(tmp_path) -> None:
    (tmp_path / "service.py").write_text(
        "def load_session(session_id):\n    return database.get(session_id)\n",
        encoding="utf-8",
    )
    index = CodeIndex(tmp_path, tmp_path / "index.sqlite")

    stats = index.rebuild()
    hits = index.search("load session")

    assert stats.files == 1
    assert stats.chunks == 1
    assert stats.updated_files == 1
    assert hits[0].path == "service.py"
    assert hits[0].start_line == 1
    assert hits[0].symbol == "load_session"


def test_code_index_updates_changed_files_and_removes_deleted_files(tmp_path) -> None:
    source = tmp_path / "service.py"
    source.write_text("def old_name():\n    return 1\n", encoding="utf-8")
    index = CodeIndex(tmp_path, tmp_path / "index.sqlite")
    index.rebuild()

    source.write_text("def new_name():\n    return 2\n", encoding="utf-8")
    updated = index.rebuild()
    assert updated.updated_files == 1
    assert index.search("new_name")[0].symbol == "new_name"
    assert index.search("old_name") == []

    source.unlink()
    removed = index.rebuild()
    assert removed.removed_files == 1
    assert removed.files == 0
    assert removed.chunks == 0


def test_hybrid_search_can_find_semantic_match_without_shared_terms(tmp_path) -> None:
    (tmp_path / "billing.py").write_text(
        "def charge_card():\n    return gateway.capture()\n",
        encoding="utf-8",
    )
    index = CodeIndex(tmp_path, tmp_path / "index.sqlite", embedder=FakeEmbedder())
    index.rebuild()

    hits = index.search("payment")

    assert hits[0].path == "billing.py"
    assert index.stats().retrieval_mode == "hybrid"


def test_enabling_embeddings_reindexes_existing_lexical_chunks(tmp_path) -> None:
    (tmp_path / "billing.py").write_text("def charge_card():\n    return True\n", encoding="utf-8")
    database = tmp_path / "index.sqlite"
    CodeIndex(tmp_path, database).rebuild()

    hybrid = CodeIndex(tmp_path, database, embedder=FakeEmbedder())
    stats = hybrid.rebuild()

    assert stats.updated_files == 1
    assert hybrid.search("payment")[0].path == "billing.py"


def test_index_skips_generated_and_private_runtime_directories(tmp_path) -> None:
    (tmp_path / "main.py").write_text("print('safe')\n", encoding="utf-8")
    (tmp_path / ".vela").mkdir()
    (tmp_path / ".vela" / "secret.py").write_text("TOKEN = 'secret'\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "vendor.js").write_text("vendor()\n", encoding="utf-8")
    index = CodeIndex(tmp_path, tmp_path / "index.sqlite")

    stats = index.rebuild()

    assert stats.files == 1
    assert index.search("secret") == []
    assert index.search("vendor") == []
