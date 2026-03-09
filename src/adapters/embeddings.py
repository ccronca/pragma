"""Embedding model factory.

Selects the embedding backend based on the 'embeddings.provider' config key.
Both indexing and search must use the same provider and model so vectors
are in the same space.

Supported providers:
- gemini: Google Gemini API (requires GEMINI_API_KEY environment variable)
- local: Local HuggingFace sentence-transformers model (no API key needed)
"""

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_MODEL = "BAAI/bge-large-en-v1.5"


def get_embed_model(config: dict):
    """Return a LlamaIndex-compatible embedding model from config.

    Args:
        config: Loaded config.yaml dict.

    Returns:
        A LlamaIndex embedding model instance.

    Raises:
        ValueError: If the provider is unknown or required config is missing.
    """
    embeddings_config = config.get("embeddings", {})
    provider = embeddings_config.get("provider", "gemini")

    if provider == "gemini":
        return _make_gemini_embed_model()
    elif provider == "local":
        model_name = embeddings_config.get("model", _DEFAULT_LOCAL_MODEL)
        return _make_local_embed_model(model_name)
    else:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. "
            "Supported values: 'gemini', 'local'."
        )


def _make_gemini_embed_model(model_name: str = "gemini-embedding-001"):
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    logger.info("Using Gemini embedding model: %s", model_name)
    return GoogleGenAIEmbedding(model_name=model_name, api_key=api_key)


def _make_local_embed_model(model_name: str):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    logger.info("Using local embedding model: %s", model_name)
    return HuggingFaceEmbedding(model_name=model_name)
