import logging
from pathlib import Path

import chromadb
from filelock import FileLock
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from adapters.embeddings import get_embed_model

logger = logging.getLogger(__name__)

# Shared lock file used by the indexer (writer) and API server (reader) to prevent
# ChromaDB InternalError caused by concurrent multi-process access.
_CHROMA_LOCK_FILE = Path("./data/chroma.lock")


def index_merge_requests(config: dict, repo_configs: list[dict]) -> None:
    """Embed and store merge requests in ChromaDB.

    Each MR is indexed as two separate documents — one for the diff (code patterns)
    and one for review discussions (business reasoning) — to prevent signal dilution
    and enable targeted searches by content type.

    Args:
        config: Pragma configuration dict with embedding and vector store settings
        repo_configs: List of repository configurations to index. Each entry must include
                      owner, name, and GitLab connection info.
    """
    embed_model = get_embed_model(config)
    Settings.embed_model = embed_model
    Settings.text_splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=200)

    chroma_path = config.get("vector_store", {}).get("path", "./data/chroma_db")
    logger.info("Connecting to ChromaDB at %s", chroma_path)

    _CHROMA_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(_CHROMA_LOCK_FILE, timeout=300)

    with lock:
        logger.info("Acquired ChromaDB write lock")
        db = chromadb.PersistentClient(path=chroma_path)
        chroma_collection = db.get_or_create_collection("pragma_collection")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        all_documents = []

        for repo_config in repo_configs:
            try:
                docs = _process_repo(repo_config, config, chroma_collection)
                all_documents.extend(docs)
            except Exception as e:
                repo_id = (
                    f"{repo_config.get('owner', '?')}/{repo_config.get('name', '?')}"
                )
                logger.error("Failed to process repository %s: %s", repo_id, e)

        if not all_documents:
            logger.info(
                "No new MRs to index. Total documents in ChromaDB: %d",
                chroma_collection.count(),
            )
            return

        logger.info("Indexing %d documents for new MRs", len(all_documents))
        VectorStoreIndex.from_documents(
            all_documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

        new_mr_count = len({d.metadata["mr_id"] for d in all_documents})
        logger.info(
            "Indexed %d new MRs (%d documents). Total in ChromaDB: %d",
            new_mr_count,
            len(all_documents),
            chroma_collection.count(),
        )

    logger.info("Released ChromaDB write lock")


def _process_repo(repo_config: dict, config: dict, chroma_collection) -> list[Document]:
    """Fetch MRs from a single repository and build documents."""
    import os

    from adapters.gitlab import GitlabAdapter

    owner = repo_config["owner"]
    name = repo_config["name"]
    gitlab_config = config.get("gitlab", {})
    base_url = (
        repo_config.get("base_url")
        or gitlab_config.get("base_url")
        or os.getenv("GITLAB_BASE_URL")
        or "https://gitlab.com"
    )
    token = os.environ.get("GITLAB_PRIVATE_TOKEN")
    if not token:
        raise ValueError("GITLAB_PRIVATE_TOKEN environment variable is not set")

    max_mrs = repo_config.get("max_mrs", 50)
    state = repo_config.get("state", "merged")
    updated_after = repo_config.get("updated_after")

    logger.info("Processing repository %s/%s", owner, name)
    adapter = GitlabAdapter(
        base_url=base_url,
        private_token=token,
        owner=owner,
        name=name,
    )
    merge_requests = adapter.fetch_mrs(
        state=state, max_mrs=max_mrs, updated_after=updated_after
    )
    logger.info("Fetched %d MRs from %s/%s", len(merge_requests), owner, name)

    already_indexed = _get_indexed_mr_ids(
        chroma_collection, repo_owner=owner, repo_name=name
    )
    if already_indexed:
        logger.info(
            "Found %d already-indexed MRs for %s/%s, skipping duplicates",
            len(already_indexed),
            owner,
            name,
        )

    return _build_documents(merge_requests, already_indexed)


def _build_documents(merge_requests: list, already_indexed: set) -> list[Document]:
    """Build LlamaIndex documents from MR data, skipping already-indexed MRs.

    Creates two documents per MR:
    - diff document: for code-pattern similarity searches
    - discussion document: for business-reasoning similarity searches
    """
    documents = []

    for mr in merge_requests:
        if mr["id"] in already_indexed:
            continue

        base_metadata = {
            "mr_id": mr["id"],
            "mr_title": mr["title"],
            "mr_description": mr["description"][:500] if mr["description"] else "",
            "repo_owner": mr["repo_owner"],
            "repo_name": mr["repo_name"],
            "author": mr.get("author"),
            "created_at": mr.get("created_at"),
            "merged_at": mr.get("merged_at"),
            "web_url": mr.get("web_url"),
        }

        if mr.get("diff"):
            documents.append(
                Document(
                    text=(
                        f"MR Title: {mr['title']}\n"
                        f"MR Description: {mr['description']}\n"
                        f"Code Changes (Diff):\n{mr['diff']}"
                    ),
                    metadata={**base_metadata, "content_type": "diff"},
                )
            )

        if mr.get("discussions"):
            notes = "\n".join(
                f"- {d['author']}: {d['note']}" for d in mr["discussions"]
            )
            documents.append(
                Document(
                    text=(
                        f"MR Title: {mr['title']}\n"
                        f"MR Description: {mr['description']}\n"
                        f"Review Discussion:\n{notes}"
                    ),
                    metadata={**base_metadata, "content_type": "discussion"},
                )
            )

    return documents


def _get_indexed_mr_ids(
    chroma_collection,
    repo_owner: str | None = None,
    repo_name: str | None = None,
) -> set:
    """Return the set of MR IDs already present in the collection.

    When repo_owner and repo_name are provided, only returns MR IDs
    for that specific repository.
    """
    if chroma_collection.count() == 0:
        return set()

    where_filter = None
    if repo_owner and repo_name:
        where_filter = {
            "$and": [
                {"repo_owner": {"$eq": repo_owner}},
                {"repo_name": {"$eq": repo_name}},
            ]
        }

    results = chroma_collection.get(include=["metadatas"], where=where_filter)
    return {
        m["mr_id"]
        for m in results.get("metadatas", [])
        if m and m.get("mr_id") is not None
    }
