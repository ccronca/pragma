import logging

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from adapters.embeddings import get_embed_model

logger = logging.getLogger(__name__)


def index_merge_requests(config: dict, merge_requests: list) -> None:
    """Embed and store merge requests in ChromaDB.

    Each MR is indexed as two separate documents — one for the diff (code patterns)
    and one for review discussions (business reasoning) — to prevent signal dilution
    and enable targeted searches by content type.
    """
    logger.info("Processing and embedding %d merge requests", len(merge_requests))

    embed_model = get_embed_model(config)
    Settings.embed_model = embed_model
    Settings.text_splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=200)

    chroma_path = config.get("vector_store", {}).get("path", "./data/chroma_db")
    logger.info("Connecting to ChromaDB at %s", chroma_path)
    db = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection("pragma_collection")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    already_indexed = _get_indexed_mr_ids(chroma_collection)
    if already_indexed:
        logger.info(
            "Found %d already-indexed MRs, skipping duplicates", len(already_indexed)
        )

    documents = _build_documents(merge_requests, already_indexed)
    skipped = (
        len(merge_requests) - len({d.metadata["mr_id"] for d in documents})
        if documents
        else len(merge_requests)
    )

    if skipped:
        logger.info("Skipped %d already-indexed MRs", skipped)

    if not documents:
        logger.info(
            "No new MRs to index. Total documents in ChromaDB: %d",
            chroma_collection.count(),
        )
        return

    logger.info("Indexing %d documents for new MRs", len(documents))
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    new_mr_count = len({d.metadata["mr_id"] for d in documents})
    logger.info(
        "Indexed %d new MRs (%d documents). Total in ChromaDB: %d",
        new_mr_count,
        len(documents),
        chroma_collection.count(),
    )


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


def _get_indexed_mr_ids(chroma_collection) -> set:
    """Return the set of MR IDs already present in the collection."""
    if chroma_collection.count() == 0:
        return set()

    results = chroma_collection.get(include=["metadatas"])
    return {
        m["mr_id"]
        for m in results.get("metadatas", [])
        if m and m.get("mr_id") is not None
    }
