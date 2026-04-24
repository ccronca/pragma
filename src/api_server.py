#!/usr/bin/env python3
"""
Pragma REST API - Historical MR Context Provider

REST API exposing the indexed GitLab MR database for semantic search.
Any HTTP client (MCP tools, scripts, AI assistants) can query for historical context.

Endpoints:
- GET  /health       - Health check
- POST /search       - Semantic search for similar MRs
- GET  /mrs/{mr_id}  - Get details of specific MR
- GET  /mrs          - List all indexed MRs
- GET  /stats        - Database statistics
"""

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import chromadb
import chromadb.errors
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel, Field

from adapters.embeddings import get_embed_model


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
    repository: Optional[str] = Field(
        None,
        description=(
            "Filter results to a specific repository (format: 'owner/name', "
            "e.g. 'my-group/my-repo')"
        ),
    )


class MRSummary(BaseModel):
    """Summary of a merge request."""

    mr_id: int
    mr_title: str
    repo_owner: Optional[str] = None
    repo_name: Optional[str] = None
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


class ReviewSummary(BaseModel):
    """Summary of an AI-generated MR review."""

    repository: str
    mr_id: int
    updated_at: Optional[str]
    review_filename: str


class RepositoryInfo(BaseModel):
    """Information about an indexed repository."""

    owner: str
    name: str
    full_name: str
    mr_count: int


class RepositoryStats(BaseModel):
    """Per-repository statistics."""

    full_name: str
    total_documents: int
    unique_mrs: int


class DatabaseStats(BaseModel):
    """Statistics about the indexed database."""

    total_documents: int
    unique_mrs: int
    collection_name: str
    vector_store_path: str
    per_repository: list[RepositoryStats] = []


class PragmaAPI:
    """Pragma API state and operations."""

    def __init__(self):
        self.config = None
        self.index = None
        self.chroma_collection = None
        self.chroma_db = None
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

        # Initialize embedding model from config
        embed_model = get_embed_model(self.config)
        Settings.embed_model = embed_model

        chroma_path = self.config.get("vector_store", {}).get(
            "path", "./data/chroma_db"
        )

        # Resolve chroma_path relative to project root
        if not Path(chroma_path).is_absolute():
            current_dir = Path.cwd()
            if (current_dir / chroma_path).exists():
                chroma_path = str(current_dir / chroma_path)
            elif (current_dir.parent / chroma_path).exists():
                chroma_path = str(current_dir.parent / chroma_path)

        self._chroma_path = chroma_path

        # Load ChromaDB
        self.chroma_db = chromadb.PersistentClient(path=chroma_path)
        self.chroma_collection = self.chroma_db.get_or_create_collection(
            "pragma_collection"
        )
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        # Load index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )

        self.initialized = True

    def _refresh(self):
        """Reinitialize the ChromaDB client and collection after a stale reference.

        Called automatically when a NotFoundError is detected. Reinitializes the
        full PersistentClient (not just the collection) to recover from cases where
        the client's internal state has become invalid after an indexer write cycle.
        """
        self.chroma_db = chromadb.PersistentClient(path=self._chroma_path)
        self.chroma_collection = self.chroma_db.get_or_create_collection(
            "pragma_collection"
        )
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=Settings.embed_model
        )


pragma_api = PragmaAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pragma_api.initialize()
    yield


app = FastAPI(
    title="Pragma API",
    description=(
        "Historical GitLab MR database for context-aware code reviews. "
        "Search for similar merge requests to provide AI assistants with "
        "institutional knowledge from past code changes, discussions, and decisions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Exception handler for stale ChromaDB collection references.
# This happens when 'pragma clear-index' is run while the server is active:
# the collection is deleted and recreated with a new UUID, making the cached
# reference invalid. Auto-recovery refreshes the reference so the next request succeeds.
@app.exception_handler(chromadb.errors.NotFoundError)
async def chromadb_not_found_handler(request, exc):
    pragma_api._refresh()
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "Database collection was reset (likely after 'pragma clear-index'). "
                "References have been refreshed — please retry your request."
            )
        },
    )


@app.exception_handler(chromadb.errors.InternalError)
async def chromadb_internal_error_handler(request, exc):
    # Transient error caused by concurrent indexer write + API read on the same
    # ChromaDB PersistentClient. The indexer holds a file lock during writes;
    # the API may still see this if it opened a connection just before the lock
    # was acquired. Returning 503 signals the client to retry.
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "Database is temporarily unavailable (concurrent write in progress). "
                "Please retry your request in a few seconds."
            )
        },
    )


# API Endpoints


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

    # Count unique MRs and build per-repository breakdown
    results = pragma_api.chroma_collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])
    unique_mrs = len(set(m.get("mr_id") for m in metadatas if m and m.get("mr_id")))

    repo_docs: dict[str, list[dict]] = {}
    for m in metadatas:
        if not m:
            continue
        owner = m.get("repo_owner", "")
        name = m.get("repo_name", "")
        key = f"{owner}/{name}" if owner or name else ""
        repo_docs.setdefault(key, []).append(m)

    per_repository = []
    for full_name, docs in sorted(repo_docs.items()):
        if not full_name:
            continue
        repo_mr_ids = {d.get("mr_id") for d in docs if d.get("mr_id") is not None}
        per_repository.append(
            RepositoryStats(
                full_name=full_name,
                total_documents=len(docs),
                unique_mrs=len(repo_mr_ids),
            )
        )

    return DatabaseStats(
        total_documents=count,
        unique_mrs=unique_mrs,
        collection_name="pragma_collection",
        vector_store_path=pragma_api.config.get("vector_store", {}).get(
            "path", "./data/chroma_db"
        ),
        per_repository=per_repository,
    )


@app.post("/search", response_model=list[MRSearchResult])
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

    # Push filters into the vector store so retrieval only considers
    # matching documents, instead of filtering client-side after retrieval.
    metadata_filters = []
    if request.content_type:
        metadata_filters.append(
            MetadataFilter(key="content_type", value=request.content_type)
        )
    if request.repository:
        parts = request.repository.split("/", 1)
        if len(parts) == 2:
            metadata_filters.append(MetadataFilter(key="repo_owner", value=parts[0]))
            metadata_filters.append(MetadataFilter(key="repo_name", value=parts[1]))
    filters = MetadataFilters(filters=metadata_filters) if metadata_filters else None

    retriever = pragma_api.index.as_retriever(
        similarity_top_k=request.top_k, filters=filters
    )
    nodes = retriever.retrieve(query_text)

    results = []
    for node in nodes:
        if node.score < request.min_score:
            continue

        metadata = node.metadata or {}
        node_content_type = metadata.get("content_type", "diff")

        results.append(
            MRSearchResult(
                mr_id=metadata.get("mr_id", 0),
                mr_title=metadata.get("mr_title", ""),
                repo_owner=metadata.get("repo_owner"),
                repo_name=metadata.get("repo_name"),
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


@app.get("/repositories", response_model=list[RepositoryInfo])
async def list_repositories():
    """List all indexed repositories with MR counts."""
    if not pragma_api.initialized:
        pragma_api.initialize()

    if pragma_api.chroma_collection.count() == 0:
        return []

    results = pragma_api.chroma_collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    # Group by (repo_owner, repo_name) and count unique MRs
    repo_mrs: dict[tuple[str, str], set] = {}
    for m in metadatas:
        if not m:
            continue
        owner = m.get("repo_owner", "")
        name = m.get("repo_name", "")
        if not owner or not name:
            continue
        key = (owner, name)
        mr_id = m.get("mr_id")
        if mr_id is not None:
            repo_mrs.setdefault(key, set()).add(mr_id)

    return [
        RepositoryInfo(
            owner=owner,
            name=name,
            full_name=f"{owner}/{name}",
            mr_count=len(mr_ids),
        )
        for (owner, name), mr_ids in sorted(repo_mrs.items())
    ]


@app.get("/mrs/{mr_id}", response_model=MRDetails)
async def get_mr_details(mr_id: int):
    """
    Get full details of a specific merge request by IID.

    Returns complete MR content including description, diff, and discussions.

    Example usage (from Python sandbox):
    ```python
    import requests
    response = requests.get("http://localhost:8000/mrs/42")
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
        repo_owner=metadata.get("repo_owner"),
        repo_name=metadata.get("repo_name"),
        mr_description=metadata.get("mr_description", ""),
        author=metadata.get("author"),
        created_at=metadata.get("created_at"),
        merged_at=metadata.get("merged_at"),
        web_url=metadata.get("web_url"),
        full_content=results["documents"][0],
    )


@app.get("/mrs", response_model=list[MRSummary])
async def list_indexed_mrs(
    limit: int = Query(
        50, ge=1, le=200, description="Maximum number of unique MRs to return"
    ),
    offset: int = Query(
        0, ge=0, description="Offset for pagination (based on unique MRs)"
    ),
    repository: Optional[str] = Query(
        None,
        description=(
            "Filter by repository (format: 'owner/name', e.g. 'my-group/my-repo')"
        ),
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

    # Build optional repository filter
    where_filter = None
    if repository:
        parts = repository.split("/", 1)
        if len(parts) == 2:
            where_filter = {
                "$and": [
                    {"repo_owner": {"$eq": parts[0]}},
                    {"repo_name": {"$eq": parts[1]}},
                ]
            }

    # Get documents from ChromaDB
    results = pragma_api.chroma_collection.get(
        limit=fetch_limit, include=["metadatas"], where=where_filter
    )

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
                    repo_owner=metadata.get("repo_owner"),
                    repo_name=metadata.get("repo_name"),
                    author=metadata.get("author"),
                    created_at=metadata.get("created_at"),
                    merged_at=metadata.get("merged_at"),
                    web_url=metadata.get("web_url"),
                )
            )

    # Apply offset and limit to unique MRs
    paginated_mrs = all_unique_mrs[offset : offset + limit]

    return paginated_mrs


def _resolve_data_dir() -> Path:
    """Return the project data/ directory regardless of working directory."""
    current = Path.cwd()
    for candidate in (current / "data", current.parent / "data"):
        if candidate.is_dir():
            return candidate
    return current / "data"


_REVIEW_FILENAME_RE = re.compile(r"^(.+)_mr(\d+)_\d{8}_\d{6}\.md$")


def _reviews_from_filesystem(
    reviews_dir: Path, known_repos: dict[str, str]
) -> list[ReviewSummary]:
    """Build ReviewSummary list by scanning the reviews directory.

    Used as a fallback when the state file uses the legacy reviewed_mr_ids format
    that does not track file paths. known_repos maps repo_name -> repo_key.
    """
    summaries = []
    for path in reviews_dir.glob("*.md"):
        m = _REVIEW_FILENAME_RE.match(path.name)
        if not m:
            continue
        repo_name, mr_id = m.group(1), int(m.group(2))
        repo_key = known_repos.get(repo_name, repo_name)
        summaries.append(
            ReviewSummary(
                repository=repo_key,
                mr_id=mr_id,
                updated_at=None,
                review_filename=path.name,
            )
        )
    return summaries


@app.get("/reviews", response_model=list[ReviewSummary])
async def list_reviews(
    repository: Optional[str] = Query(
        None, description="Filter by repository (format: 'owner/name')"
    ),
):
    """List all AI-generated MR reviews saved by the reviewer agent."""
    data_dir = _resolve_data_dir()
    reviews_dir = data_dir / "reviews"
    state_file = data_dir / "review_state.json"

    state = {}
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)

    # Build a map from short repo name -> full repo key for filesystem fallback
    known_repos: dict[str, str] = {}
    for repo_key in state.get("repositories", {}):
        short_name = repo_key.rsplit("/", 1)[-1]
        known_repos[short_name] = repo_key

    reviews: list[ReviewSummary] = []
    seen_filenames: set[str] = set()

    # Prefer structured state (new format with file paths)
    for repo_key, repo_state in state.get("repositories", {}).items():
        if repository and repo_key != repository:
            continue
        for mr_id, info in repo_state.get("reviewed_mrs", {}).items():
            if not info.get("review_file"):
                continue
            review_path = Path(info["review_file"])
            seen_filenames.add(review_path.name)
            reviews.append(
                ReviewSummary(
                    repository=repo_key,
                    mr_id=int(mr_id),
                    updated_at=info.get("updated_at"),
                    review_filename=review_path.name,
                )
            )

    # Fall back to filesystem scan for files not tracked in state (legacy format)
    if reviews_dir.exists():
        for summary in _reviews_from_filesystem(reviews_dir, known_repos):
            if summary.review_filename in seen_filenames:
                continue
            if repository and summary.repository != repository:
                continue
            reviews.append(summary)

    return sorted(reviews, key=lambda r: r.review_filename, reverse=True)


@app.get("/reviews/{review_filename}")
async def get_review(review_filename: str):
    """Get the full Markdown content of a specific review file."""
    data_dir = _resolve_data_dir()
    review_path = data_dir / "reviews" / review_filename

    # Guard against path traversal
    try:
        review_path.resolve().relative_to((data_dir / "reviews").resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid review filename")

    if not review_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Review not found: {review_filename}"
        )

    return {
        "filename": review_filename,
        "content": review_path.read_text(encoding="utf-8"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
