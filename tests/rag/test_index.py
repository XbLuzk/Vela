from __future__ import annotations

import stat

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


def test_index_does_not_follow_file_or_directory_symlinks(tmp_path) -> None:
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    (repository / "main.py").write_text("SAFE_MARKER = True\n", encoding="utf-8")
    (outside / "secret.py").write_text("OUTSIDE_SECRET = True\n", encoding="utf-8")
    (repository / "linked.py").symlink_to(outside / "secret.py")
    (repository / "linked-dir").symlink_to(outside, target_is_directory=True)

    index = CodeIndex(repository, tmp_path / "index.sqlite")
    stats = index.rebuild()

    assert stats.files == 1
    assert index.search("OUTSIDE_SECRET") == []


def test_repository_root_named_like_excluded_directory_is_still_indexed(tmp_path) -> None:
    repository = tmp_path / "node_modules"
    repository.mkdir()
    (repository / "main.py").write_text("ROOT_SOURCE = True\n", encoding="utf-8")

    index = CodeIndex(repository, tmp_path / "index.sqlite")

    assert index.rebuild().files == 1


def test_embedding_outage_falls_back_to_lexical_index_and_search(tmp_path) -> None:
    class BrokenEmbedder:
        identity = "broken"

        def embed(self, texts):  # noqa: ARG002
            raise OSError("offline")

    (tmp_path / "service.py").write_text("def recover_session(): return True\n", encoding="utf-8")
    index = CodeIndex(tmp_path, tmp_path / "index.sqlite", embedder=BrokenEmbedder())

    stats = index.rebuild()
    hits = index.search("recover_session")

    assert stats.files == 1
    assert hits[0].path == "service.py"
    assert "unavailable" in str(index.last_warning)


def test_rebuild_batches_embeddings_across_changed_files(tmp_path) -> None:
    class CountingEmbedder(FakeEmbedder):
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed(self, texts):
            self.calls.append(texts)
            return super().embed(texts)

    for index in range(3):
        (tmp_path / f"service_{index}.py").write_text(
            f"def service_{index}(): return 'payment'\n",
            encoding="utf-8",
        )
    embedder = CountingEmbedder()

    CodeIndex(tmp_path, tmp_path / "index.sqlite", embedder=embedder).rebuild()

    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 3


def test_index_database_is_private_to_the_current_user(tmp_path) -> None:
    database = tmp_path / "data" / "index.sqlite"
    CodeIndex(tmp_path, database)

    assert stat.S_IMODE(database.stat().st_mode) == 0o600
    assert stat.S_IMODE(database.parent.stat().st_mode) == 0o700
