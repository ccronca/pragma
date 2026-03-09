"""Unit tests for the embedding model factory."""

from unittest.mock import MagicMock, patch

import pytest

from adapters.embeddings import get_embed_model


class TestGetEmbedModelProvider:
    def test_defaults_to_gemini_when_no_embeddings_section(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "llama_index.embeddings.google_genai.GoogleGenAIEmbedding"
            ) as mock_cls:
                mock_cls.return_value = MagicMock()
                get_embed_model({})
                mock_cls.assert_called_once()

    def test_selects_gemini_provider_explicitly(self):
        config = {"embeddings": {"provider": "gemini"}}
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "llama_index.embeddings.google_genai.GoogleGenAIEmbedding"
            ) as mock_cls:
                mock_cls.return_value = MagicMock()
                get_embed_model(config)
                mock_cls.assert_called_once()

    def test_selects_local_provider(self):
        config = {"embeddings": {"provider": "local"}}
        with patch(
            "llama_index.embeddings.huggingface.HuggingFaceEmbedding"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            get_embed_model(config)
            mock_cls.assert_called_once()

    def test_raises_for_unknown_provider(self):
        config = {"embeddings": {"provider": "openai"}}
        with pytest.raises(ValueError, match="Unknown embedding provider 'openai'"):
            get_embed_model(config)


class TestGeminiProvider:
    def test_raises_when_api_key_not_set(self):
        with patch.dict("os.environ", {}, clear=True):
            # Ensure GEMINI_API_KEY is absent
            import os

            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                get_embed_model({"embeddings": {"provider": "gemini"}})

    def test_passes_api_key_to_model(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "my-secret-key"}):
            with patch(
                "llama_index.embeddings.google_genai.GoogleGenAIEmbedding"
            ) as mock_cls:
                mock_cls.return_value = MagicMock()
                get_embed_model({"embeddings": {"provider": "gemini"}})
                _, kwargs = mock_cls.call_args
                assert kwargs["api_key"] == "my-secret-key"


class TestLocalProvider:
    def test_uses_default_model_when_not_specified(self):
        config = {"embeddings": {"provider": "local"}}
        with patch(
            "llama_index.embeddings.huggingface.HuggingFaceEmbedding"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            get_embed_model(config)
            _, kwargs = mock_cls.call_args
            assert kwargs["model_name"] == "BAAI/bge-large-en-v1.5"

    def test_uses_configured_model_name(self):
        config = {"embeddings": {"provider": "local", "model": "all-mpnet-base-v2"}}
        with patch(
            "llama_index.embeddings.huggingface.HuggingFaceEmbedding"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            get_embed_model(config)
            _, kwargs = mock_cls.call_args
            assert kwargs["model_name"] == "all-mpnet-base-v2"
