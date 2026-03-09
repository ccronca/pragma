import typer
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
import chromadb
import os


def index_merge_requests(config: dict, merge_requests: list):
    typer.echo("Processing and embedding merge requests...")

    # 1. Initialize Gemini Embedding Model
    gemini_api_key = config.get("api_keys", {}).get("gemini")
    if gemini_api_key == "YOUR_GEMINI_API_KEY" or not gemini_api_key:
        typer.echo("Gemini API key not configured. Please run 'pragma init'.", err=True)
        raise typer.Exit(code=1)

    os.environ["GEMINI_API_KEY"] = gemini_api_key
    embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")
    Settings.embed_model = embed_model
    Settings.text_splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=200)

    # 2. Initialize ChromaDB Vector Store
    chroma_path = config.get("vector_store", {}).get("path", "./data/chroma_db")
    typer.echo(f"Connecting to ChromaDB at {chroma_path}...")
    db = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection("pragma_collection")

    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Skip already-indexed MRs
    already_indexed = _get_indexed_mr_ids(chroma_collection)
    if already_indexed:
        typer.echo(
            f"Found {len(already_indexed)} already-indexed MRs, skipping duplicates."
        )

    # 4. Build separate documents per content type for each new MR
    #
    # Why separate documents?
    # - Diffs are large and dominate the embedding when mixed with discussions
    # - Discussions contain business logic and decisions ("why") which should be
    #   retrievable independently from code changes ("what")
    # - Separate embeddings let the LLM find relevant reasoning even when the
    #   code pattern doesn't match closely
    documents = []
    skipped = 0

    for mr in merge_requests:
        if mr["id"] in already_indexed:
            skipped += 1
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

        # Document 1: Code changes (diff)
        # Good for: "find MRs with similar code changes / patterns"
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

        # Document 2: Discussions and review notes
        # Good for: "find MRs where team discussed X / decided Y"
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

    if skipped:
        typer.echo(f"Skipped {skipped} already-indexed MRs.")

    if not documents:
        typer.echo("No new MRs to index.")
        typer.echo(f"Total documents in ChromaDB: {chroma_collection.count()}")
        return

    typer.echo(f"Indexing {len(documents)} documents for new MRs...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    new_mr_count = len({d.metadata["mr_id"] for d in documents})
    typer.echo(
        f"Successfully indexed {new_mr_count} new MRs ({len(documents)} documents)."
    )
    typer.echo(f"Total documents in ChromaDB: {chroma_collection.count()}")


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
