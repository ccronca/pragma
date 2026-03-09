#!/usr/bin/env python3
"""
Pragma REST API - Historical MR Context Provider

A simple REST API that exposes the indexed GitLab MR database for semantic search.
MCP clients (via Python sandbox) can call this API to get historical context.

API Design:
- Simple REST endpoints with clear documentation
- OpenAPI/Swagger for discoverability
- Callable from MCP Python sandbox (like Monty)

Endpoints:
- GET  /health                                 - Health check
- POST /search                                 - Semantic search for similar MRs
- GET  /mrs/{mr_id}                           - Get details of specific MR
- GET  /mrs                                    - List all indexed MRs
- GET  /stats                                  - Database statistics
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
import yaml
from pathlib import Path

# LlamaIndex imports
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


# Pydantic models for API
class SearchRequest(BaseModel):
    """Request for searching similar MRs.

    Use `query` for natural language searches (best for finding discussions and past decisions).
    Use `code_diff` for code-pattern searches (best for finding similar implementations).
    Exactly one of `query` or `code_diff` must be provided.
    Use `content_type` to restrict results to a specific document type.
    """

    query: Optional[str] = Field(
        None,
        description="Natural language query (best for finding discussions and past decisions)",
    )
    code_diff: Optional[str] = Field(
        None,
        description="Code diff to search for (best for finding similar code patterns)",
    )
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)
    min_score: float = Field(
        0.0, description="Minimum similarity score (0-1)", ge=0.0, le=1.0
    )
    content_type: Optional[str] = Field(
        None, description="Filter by content type: 'diff' or 'discussion'"
    )


class MRSummary(BaseModel):
    """Summary of a merge request."""

    mr_id: int
    mr_title: str
    author: Optional[str]
    created_at: Optional[str]
    merged_at: Optional[str]
    web_url: Optional[str]


class MRSearchResult(MRSummary):
    """Search result with similarity score."""

    similarity_score: float
    content_type: str  # "diff" or "discussion" - indicates what matched
    mr_description_preview: str
    content_preview: str


class MRDetails(MRSummary):
    """Full details of a merge request."""

    mr_description: str
    full_content: str


class DatabaseStats(BaseModel):
    """Statistics about the indexed database."""

    total_documents: int
    unique_mrs: int
    collection_name: str
    vector_store_path: str


# FastAPI app
app = FastAPI(
    title="Pragma API",
    description=(
        "Historical GitLab MR database for context-aware code reviews. "
        "Search for similar merge requests to provide AI assistants with "
        "institutional knowledge from past code changes, discussions, and decisions."
    ),
    version="1.0.0",
)


class PragmaAPI:
    """Pragma API state and operations."""

    def __init__(self):
        self.config = None
        self.index = None
        self.chroma_collection = None
        self.initialized = False

    def load_config(self):
        """Load configuration from config.yaml in project root."""
        # Try to find config.yaml - check parent directory if running from src/
        current_dir = Path.cwd()
        config_candidates = [
            current_dir / "config.yaml",  # Running from project root
            current_dir.parent / "config.yaml",  # Running from src/
        ]

        config_file = None
        for candidate in config_candidates:
            if candidate.exists():
                config_file = candidate
                break

        if config_file is None:
            raise FileNotFoundError(
                "config.yaml not found. Please run 'pragma init' to create it."
            )

        with open(config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def initialize(self):
        """Initialize the vector store index."""
        if self.initialized:
            return

        if self.config is None:
            self.load_config()

        # Get configuration
        gemini_api_key = self.config.get("api_keys", {}).get("gemini")
        chroma_path = self.config.get("vector_store", {}).get(
            "path", "./data/chroma_db"
        )

        if not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY":
            raise ValueError("Gemini API key not configured in config.yaml")

        # Resolve chroma_path relative to project root
        if not Path(chroma_path).is_absolute():
            # Try both current directory and parent directory
            current_dir = Path.cwd()
            if (current_dir / chroma_path).exists():
                chroma_path = str(current_dir / chroma_path)
            elif (current_dir.parent / chroma_path).exists():
                chroma_path = str(current_dir.parent / chroma_path)

        # Initialize embedding model
        embed_model = GoogleGenAIEmbedding(
            model_name="gemini-embedding-001", api_key=gemini_api_key
        )
        Settings.embed_model = embed_model

        # Load ChromaDB
        db = chromadb.PersistentClient(path=chroma_path)
        self.chroma_collection = db.get_or_create_collection("pragma_collection")
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        # Load index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )

        self.initialized = True


# Global API instance
pragma_api = PragmaAPI()


# API Endpoints


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    pragma_api.initialize()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "initialized": pragma_api.initialized,
        "service": "Pragma Historical MR API",
    }


@app.get("/stats", response_model=DatabaseStats)
async def get_stats():
    """Get database statistics."""
    if not pragma_api.initialized:
        pragma_api.initialize()

    # Get collection info
    count = pragma_api.chroma_collection.count()

    # Count unique MRs
    results = pragma_api.chroma_collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])
    unique_mrs = len(set(m.get("mr_id") for m in metadatas if m and m.get("mr_id")))

    return DatabaseStats(
        total_documents=count,
        unique_mrs=unique_mrs,
        collection_name="pragma_collection",
        vector_store_path=pragma_api.config.get("vector_store", {}).get(
            "path", "./data/chroma_db"
        ),
    )


@app.post("/search", response_model=List[MRSearchResult])
async def search_similar_mrs(request: SearchRequest):
    """
    Search for similar merge requests using semantic search.

    LLMs should make two calls to get the best context:

    1. Search discussions with a natural language description of the change:
    ```python
    requests.post("/search", json={
        "query": "unconditional Jira API fetch without pagination",
        "content_type": "discussion",
        "top_k": 5
    })
    ```

    2. Search diffs with the actual code change:
    ```python
    requests.post("/search", json={
        "code_diff": "<the unified diff>",
        "content_type": "diff",
        "top_k": 5
    })
    ```

    Combining both results gives the LLM similar code patterns AND the team's
    past reasoning and decisions, even from unrelated MRs.
    """
    if not pragma_api.initialized:
        pragma_api.initialize()

    if pragma_api.chroma_collection.count() == 0:
        return []

    if not request.query and not request.code_diff:
        raise HTTPException(
            status_code=422, detail="Provide either 'query' or 'code_diff'."
        )

    # Use the appropriate query text for the embedding
    query_text = request.query or request.code_diff

    # Fetch more results than needed so we can filter by content_type
    fetch_k = request.top_k * 3 if request.content_type else request.top_k
    retriever = pragma_api.index.as_retriever(similarity_top_k=fetch_k)
    nodes = retriever.retrieve(query_text)

    results = []
    for node in nodes:
        if node.score < request.min_score:
            continue

        metadata = node.metadata or {}
        node_content_type = metadata.get("content_type", "diff")

        # Apply content_type filter if requested
        if request.content_type and node_content_type != request.content_type:
            continue

        results.append(
            MRSearchResult(
                mr_id=metadata.get("mr_id", 0),
                mr_title=metadata.get("mr_title", ""),
                content_type=node_content_type,
                mr_description_preview=metadata.get("mr_description", "")[:300],
                author=metadata.get("author"),
                created_at=metadata.get("created_at"),
                merged_at=metadata.get("merged_at"),
                web_url=metadata.get("web_url"),
                similarity_score=node.score,
                content_preview=node.text[:800] if node.text else "",
            )
        )

        if len(results) == request.top_k:
            break

    return results


@app.get("/mrs/{mr_id}", response_model=MRDetails)
async def get_mr_details(mr_id: int):
    """
    Get full details of a specific merge request by IID.

    Returns complete MR content including description, diff, and discussions.

    Example usage (from Python sandbox):
    ```python
    import requests
    response = requests.get("http://localhost:8000/mrs/383")
    mr = response.json()
    ```
    """
    if not pragma_api.initialized:
        pragma_api.initialize()

    # Query ChromaDB for specific MR
    results = pragma_api.chroma_collection.get(
        where={"mr_id": mr_id}, include=["metadatas", "documents"]
    )

    if not results["ids"]:
        raise HTTPException(status_code=404, detail=f"MR !{mr_id} not found in index")

    metadata = results["metadatas"][0]
    return MRDetails(
        mr_id=metadata.get("mr_id", 0),
        mr_title=metadata.get("mr_title", ""),
        mr_description=metadata.get("mr_description", ""),
        author=metadata.get("author"),
        created_at=metadata.get("created_at"),
        merged_at=metadata.get("merged_at"),
        web_url=metadata.get("web_url"),
        full_content=results["documents"][0],
    )


@app.get("/mrs", response_model=List[MRSummary])
async def list_indexed_mrs(
    limit: int = Query(
        50, ge=1, le=200, description="Maximum number of unique MRs to return"
    ),
    offset: int = Query(
        0, ge=0, description="Offset for pagination (based on unique MRs)"
    ),
):
    """
    List all indexed merge requests with basic metadata.

    Returns unique MRs, deduplicated across all document chunks.

    Example usage (from Python sandbox):
    ```python
    import requests
    response = requests.get("http://localhost:8000/mrs?limit=10&offset=0")
    mrs = response.json()
    ```
    """
    if not pragma_api.initialized:
        pragma_api.initialize()

    # Check if collection is empty
    if pragma_api.chroma_collection.count() == 0:
        return []  # Return empty list if no MRs indexed yet

    # Fetch more documents than needed since each MR has multiple chunks
    # Multiply by 10 as a heuristic (each MR might have ~10 chunks on average)
    fetch_limit = (limit + offset) * 10

    # Get documents from ChromaDB
    results = pragma_api.chroma_collection.get(limit=fetch_limit, include=["metadatas"])

    # Deduplicate by MR ID and collect unique MRs
    seen_mr_ids = set()
    all_unique_mrs = []

    metadatas = results.get("metadatas", [])
    for metadata in metadatas:
        mr_id = metadata.get("mr_id")
        if mr_id and mr_id not in seen_mr_ids:
            seen_mr_ids.add(mr_id)
            all_unique_mrs.append(
                MRSummary(
                    mr_id=mr_id,
                    mr_title=metadata.get("mr_title", ""),
                    author=metadata.get("author"),
                    created_at=metadata.get("created_at"),
                    merged_at=metadata.get("merged_at"),
                    web_url=metadata.get("web_url"),
                )
            )

    # Apply offset and limit to unique MRs
    paginated_mrs = all_unique_mrs[offset : offset + limit]

    return paginated_mrs


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
