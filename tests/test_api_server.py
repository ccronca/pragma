"""Unit tests for the Pragma REST API."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with a mocked, initialized PragmaAPI."""
    with patch("api_server.PragmaAPI") as mock_class:
        mock_api = MagicMock()
        mock_api.initialized = True
        mock_api.config = {"vector_store": {"path": "./data/chroma_db"}}
        mock_class.return_value = mock_api

        from api_server import app
        import api_server

        api_server.pragma_api = mock_api

        yield TestClient(app), mock_api


class TestHealthEndpoint:
    def test_returns_healthy(self, client):
        test_client, mock_api = client
        mock_api.initialized = True
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestSearchEndpoint:
    def test_missing_query_and_diff_returns_422(self, client):
        test_client, _ = client
        response = test_client.post("/search", json={"top_k": 5})
        assert response.status_code == 422

    def test_search_with_query_returns_results(self, client):
        test_client, mock_api = client
        mock_api.chroma_collection.count.return_value = 10

        mock_node = MagicMock()
        mock_node.score = 0.8
        mock_node.metadata = {
            "mr_id": 42,
            "mr_title": "feat: retry logic",
            "content_type": "discussion",
            "mr_description": "Adds backoff",
            "author": "alice",
            "created_at": "2024-01-01",
            "merged_at": "2024-01-02",
            "web_url": "https://gitlab.example.com/mr/42",
        }
        mock_node.text = (
            "MR Title: feat: retry logic\nReview Discussion:\n- alice: LGTM"
        )
        mock_api.index.as_retriever.return_value.retrieve.return_value = [mock_node]

        response = test_client.post(
            "/search",
            json={"query": "retry logic", "content_type": "discussion", "top_k": 3},
        )
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 1
        assert results[0]["mr_id"] == 42
        assert results[0]["similarity_score"] == pytest.approx(0.8)

    def test_empty_collection_returns_empty_list(self, client):
        test_client, mock_api = client
        mock_api.chroma_collection.count.return_value = 0
        response = test_client.post("/search", json={"query": "something"})
        assert response.status_code == 200
        assert response.json() == []

    def test_min_score_filters_low_results(self, client):
        test_client, mock_api = client
        mock_api.chroma_collection.count.return_value = 5

        mock_node = MagicMock()
        mock_node.score = 0.2
        mock_node.metadata = {
            "mr_id": 1,
            "mr_title": "some mr",
            "content_type": "diff",
            "mr_description": "",
            "author": None,
            "created_at": None,
            "merged_at": None,
            "web_url": None,
        }
        mock_node.text = "content"
        mock_api.index.as_retriever.return_value.retrieve.return_value = [mock_node]

        response = test_client.post(
            "/search", json={"query": "something", "min_score": 0.5}
        )
        assert response.status_code == 200
        assert response.json() == []


class TestListMrsEndpoint:
    def test_empty_collection_returns_empty_list(self, client):
        test_client, mock_api = client
        mock_api.chroma_collection.count.return_value = 0
        response = test_client.get("/mrs")
        assert response.status_code == 200
        assert response.json() == []

    def test_deduplicates_by_mr_id(self, client):
        test_client, mock_api = client
        mock_api.chroma_collection.count.return_value = 3
        mock_api.chroma_collection.get.return_value = {
            "metadatas": [
                {
                    "mr_id": 1,
                    "mr_title": "MR One",
                    "author": "alice",
                    "created_at": None,
                    "merged_at": None,
                    "web_url": None,
                },
                {
                    "mr_id": 1,
                    "mr_title": "MR One",
                    "author": "alice",
                    "created_at": None,
                    "merged_at": None,
                    "web_url": None,
                },
                {
                    "mr_id": 2,
                    "mr_title": "MR Two",
                    "author": "bob",
                    "created_at": None,
                    "merged_at": None,
                    "web_url": None,
                },
            ]
        }
        response = test_client.get("/mrs")
        assert response.status_code == 200
        assert len(response.json()) == 2


class TestGetMrEndpoint:
    def test_returns_mr_details(self, client):
        test_client, mock_api = client
        mock_api.chroma_collection.get.return_value = {
            "ids": ["doc-1"],
            "metadatas": [
                {
                    "mr_id": 42,
                    "mr_title": "feat: retry",
                    "mr_description": "desc",
                    "author": "alice",
                    "created_at": "2024-01-01",
                    "merged_at": "2024-01-02",
                    "web_url": "https://gitlab.example.com/mr/42",
                }
            ],
            "documents": ["full content here"],
        }
        response = test_client.get("/mrs/42")
        assert response.status_code == 200
        data = response.json()
        assert data["mr_id"] == 42
        assert data["full_content"] == "full content here"

    def test_returns_404_for_missing_mr(self, client):
        test_client, mock_api = client
        mock_api.chroma_collection.get.return_value = {
            "ids": [],
            "metadatas": [],
            "documents": [],
        }
        response = test_client.get("/mrs/999")
        assert response.status_code == 404
