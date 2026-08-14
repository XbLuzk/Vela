"""stdio MCP entrypoint for project-local code retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from vela_rag.embedding import embedding_client_from_env
from vela_rag.index import CodeIndex, default_database


def create_server(root: Path) -> FastMCP:
    root = root.resolve()
    index = CodeIndex(
        root,
        default_database(root),
        embedder=embedding_client_from_env(),
    )
    server = FastMCP("Vela Code RAG", log_level="ERROR")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search_code(query: str, limit: int = 8) -> dict[str, object]:
        """Update the local index, then search code with file and line references."""
        safe_limit = min(max(limit, 1), 20)
        index.rebuild()
        rebuild_warning = index.last_warning
        result: dict[str, object] = {
            "query": query,
            "results": [hit.to_dict() for hit in index.search(query, limit=safe_limit)],
        }
        warning = index.last_warning or rebuild_warning
        if warning:
            result["warning"] = warning
        return result

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def rag_status() -> dict[str, str | int]:
        """Return index location, size, and lexical or hybrid retrieval mode."""
        return index.stats().to_dict()

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Vela project-local Code RAG MCP server")
    parser.add_argument("--root", type=Path, required=True, help="Repository root")
    args = parser.parse_args()
    create_server(args.root).run(transport="stdio")


if __name__ == "__main__":
    main()
