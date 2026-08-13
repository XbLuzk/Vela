"""Optional OpenAI-compatible embeddings for hybrid code retrieval."""

from __future__ import annotations

import os
from typing import Protocol

import httpx


class EmbeddingClient(Protocol):
    @property
    def identity(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingClient:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @property
    def identity(self) -> str:
        return f"{self.base_url}|{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json().get("data") or []
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        return [[float(value) for value in item["embedding"]] for item in ordered]


def embedding_client_from_env() -> EmbeddingClient | None:
    api_key = os.environ.get("VELA_RAG_EMBEDDING_API_KEY", "").strip()
    model = os.environ.get("VELA_RAG_EMBEDDING_MODEL", "").strip()
    if not api_key or not model:
        return None
    base_url = os.environ.get("VELA_RAG_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    return OpenAIEmbeddingClient(api_key=api_key, base_url=base_url, model=model)
