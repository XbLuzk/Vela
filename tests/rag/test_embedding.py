from __future__ import annotations

from vela_rag.embedding import OpenAIEmbeddingClient, embedding_client_from_env


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }


def test_openai_embedding_client_sends_one_ordered_batch(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return _Response()

    monkeypatch.setattr("vela_rag.embedding.httpx.post", fake_post)
    client = OpenAIEmbeddingClient(
        api_key="secret",
        base_url="https://embedding.example/v1/",
        model="embed-model",
    )

    result = client.embed(["first", "second"])

    assert result == [[1.0, 0.0], [0.0, 1.0]]
    assert captured["url"] == "https://embedding.example/v1/embeddings"
    assert captured["json"] == {"model": "embed-model", "input": ["first", "second"]}
    assert captured["headers"] == {"Authorization": "Bearer secret"}


def test_embedding_client_from_environment_requires_key_and_model(monkeypatch) -> None:
    monkeypatch.delenv("VELA_RAG_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("VELA_RAG_EMBEDDING_MODEL", raising=False)
    assert embedding_client_from_env() is None

    monkeypatch.setenv("VELA_RAG_EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv("VELA_RAG_EMBEDDING_MODEL", "embed-model")
    monkeypatch.setenv("VELA_RAG_EMBEDDING_BASE_URL", "https://embedding.example/v1")

    client = embedding_client_from_env()

    assert isinstance(client, OpenAIEmbeddingClient)
    assert client.identity == "https://embedding.example/v1|embed-model"
