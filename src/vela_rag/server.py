"""stdio MCP entrypoint for project-local code retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from vela_rag.embedding import embedding_client_from_env
from vela_rag.index import CodeIndex, default_database


def create_server(root: Path, database: Path | None = None) -> FastMCP:
    root = root.resolve()
    index = CodeIndex(
        root,
        database or default_database(root),
        embedder=embedding_client_from_env(),
    )
    server = FastMCP("Vela Code RAG", log_level="ERROR")

    @server.tool()
    def index_repository() -> dict[str, str | int]:
        """Incrementally index supported source and documentation files."""
        return index.rebuild().to_dict()

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search_code(query: str, limit: int = 8) -> dict[str, object]:
        """Search indexed code and return content with file and line references."""
        safe_limit = min(max(limit, 1), 20)
        return {
            "query": query,
            "results": [hit.to_dict() for hit in index.search(query, limit=safe_limit)],
        }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def rag_status() -> dict[str, str | int]:
        """Return index location, size, and lexical or hybrid retrieval mode."""
        return index.stats().to_dict()

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Vela project-local Code RAG MCP server")
    parser.add_argument("--root", type=Path, required=True, help="Repository root")
    parser.add_argument("--database", type=Path, help="Optional SQLite index path")
    args = parser.parse_args()
    create_server(args.root, args.database).run(transport="stdio")


if __name__ == "__main__":
    main()
