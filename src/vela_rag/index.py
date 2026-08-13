"""Incremental SQLite code index with lexical and optional vector search."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

from vela_rag.chunker import chunk_file, discover_source_files, file_digest
from vela_rag.embedding import EmbeddingClient
from vela_rag.models import CodeChunk, IndexStats, SearchHit

_TOKEN_PATTERN = re.compile(r"[\w.-]+", re.UNICODE)
_RRF_K = 60


class CodeIndex:
    """Own one project-bounded code index."""

    def __init__(
        self,
        root: str | Path,
        database: str | Path,
        *,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.database = Path(database).expanduser().resolve()
        self.embedder = embedder
        if not self.root.is_dir():
            raise ValueError(f"Repository root does not exist: {self.root}")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def rebuild(self) -> IndexStats:
        """Incrementally update changed files and remove deleted files."""
        files = list(discover_source_files(self.root))
        relative_paths = {path.relative_to(self.root).as_posix() for path in files}
        with self._connect() as connection:
            known = {
                str(row["path"]): str(row["digest"])
                for row in connection.execute("SELECT path, digest FROM files")
            }
            removed = sorted(set(known) - relative_paths)
            for relative in removed:
                self._delete_file(connection, relative)

            updated = 0
            for path in files:
                relative = path.relative_to(self.root).as_posix()
                digest = self._index_digest(file_digest(path))
                if known.get(relative) == digest:
                    continue
                self._replace_file(connection, relative, digest, chunk_file(path, self.root))
                updated += 1
        return self.stats(updated_files=updated, removed_files=len(removed))

    def search(self, query: str, *, limit: int = 8) -> list[SearchHit]:
        query = query.strip()
        if not query or limit < 1:
            return []
        with self._connect() as connection:
            lexical = self._lexical_ranking(connection, query)
            semantic = self._semantic_ranking(connection, query)
            scores: dict[str, float] = {}
            for ranking in (lexical, semantic):
                for rank, chunk_id in enumerate(ranking, start=1):
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (_RRF_K + rank)
            if not scores:
                return []
            ordered_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
            placeholders = ",".join("?" for _ in ordered_ids)
            rows = connection.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",  # noqa: S608
                ordered_ids,
            ).fetchall()
            by_id = {str(row["chunk_id"]): row for row in rows}
        return [
            SearchHit(
                path=str(by_id[chunk_id]["path"]),
                start_line=int(by_id[chunk_id]["start_line"]),
                end_line=int(by_id[chunk_id]["end_line"]),
                symbol=str(by_id[chunk_id]["symbol"]),
                content=str(by_id[chunk_id]["content"]),
                score=round(scores[chunk_id], 6),
            )
            for chunk_id in ordered_ids
            if chunk_id in by_id
        ]

    def stats(self, *, updated_files: int = 0, removed_files: int = 0) -> IndexStats:
        with self._connect() as connection:
            files = int(connection.execute("SELECT COUNT(*) FROM files").fetchone()[0])
            chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        return IndexStats(
            root=str(self.root),
            database=str(self.database),
            files=files,
            chunks=chunks,
            updated_files=updated_files,
            removed_files=removed_files,
            retrieval_mode="hybrid" if self.embedder else "lexical",
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    digest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT
                );
                CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks(path);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    path,
                    symbol,
                    content,
                    tokenize = 'unicode61'
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection

    def _replace_file(
        self,
        connection: sqlite3.Connection,
        relative: str,
        digest: str,
        chunks: list[CodeChunk],
    ) -> None:
        self._delete_file(connection, relative)
        embeddings = self._embed_chunks(chunks)
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            connection.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk.chunk_id,
                    chunk.path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.symbol,
                    chunk.content,
                    json.dumps(embedding) if embedding else None,
                ),
            )
            connection.execute(
                "INSERT INTO chunks_fts(chunk_id, path, symbol, content) VALUES (?, ?, ?, ?)",
                (chunk.chunk_id, chunk.path, chunk.symbol, chunk.content),
            )
        connection.execute("INSERT INTO files(path, digest) VALUES (?, ?)", (relative, digest))

    def _delete_file(self, connection: sqlite3.Connection, relative: str) -> None:
        chunk_ids = [
            str(row[0])
            for row in connection.execute("SELECT chunk_id FROM chunks WHERE path = ?", (relative,))
        ]
        if chunk_ids:
            connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )
        connection.execute("DELETE FROM chunks WHERE path = ?", (relative,))
        connection.execute("DELETE FROM files WHERE path = ?", (relative,))

    def _embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float] | None]:
        if self.embedder is None:
            return [None] * len(chunks)
        embeddings: list[list[float]] = []
        for offset in range(0, len(chunks), 64):
            batch = chunks[offset : offset + 64]
            values = self.embedder.embed([chunk.content for chunk in batch])
            if len(values) != len(batch):
                raise RuntimeError("Embedding provider returned an unexpected result count")
            embeddings.extend(values)
        return embeddings

    def _index_digest(self, content_digest: str) -> str:
        retrieval_identity = self.embedder.identity if self.embedder else "lexical"
        value = f"{content_digest}|{retrieval_identity}"
        return hashlib.sha256(value.encode()).hexdigest()

    def _lexical_ranking(self, connection: sqlite3.Connection, query: str) -> list[str]:
        terms = _TOKEN_PATTERN.findall(query)[:12]
        if not terms:
            return []
        expression = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        rows = connection.execute(
            "SELECT chunk_id FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT 100",
            (expression,),
        ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    def _semantic_ranking(self, connection: sqlite3.Connection, query: str) -> list[str]:
        if self.embedder is None:
            return []
        query_vectors = self.embedder.embed([query])
        if len(query_vectors) != 1:
            raise RuntimeError("Embedding provider did not return a query vector")
        query_vector = query_vectors[0]
        scored: list[tuple[float, str]] = []
        for row in connection.execute(
            "SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL"
        ):
            vector = json.loads(str(row["embedding"]))
            scored.append((_cosine(query_vector, vector), str(row["chunk_id"])))
        scored.sort(reverse=True)
        return [chunk_id for _, chunk_id in scored[:100]]


def default_database(root: Path) -> Path:
    digest = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
    return Path.home() / ".vela" / "rag" / digest / "index.sqlite"


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0
