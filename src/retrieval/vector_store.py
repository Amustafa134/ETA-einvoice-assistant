"""
Qdrant storage: create the collection, upload chunks, and search.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.ingestion.chunker import Chunk
from src.retrieval.embedder import embed_passages, embed_query

COLLECTION_NAME = "eta_einvoice"
EMBEDDING_DIM = 768  # multilingual-e5-base output size -- update if the model changes

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_collection(client: QdrantClient, recreate: bool = False):
    """Create the collection if it doesn't exist (or force-recreate it)."""
    exists = client.collection_exists(COLLECTION_NAME)
    if exists and not recreate:
        print(f"Collection '{COLLECTION_NAME}' already exists -- leaving it as is.")
        return
    if exists and recreate:
        print(f"Recreating collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}'.")


def upload_chunks(client: QdrantClient, chunks: list[Chunk], batch_size: int = 32):
    """Embed and upload chunks to Qdrant, in batches."""
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c.text for c in batch]
        vectors = embed_passages(texts)

        points = [
            PointStruct(
                id=start + i,  # simple incrementing integer id
                vector=vectors[i],
                payload={
                    "chunk_id": batch[i].id,
                    "text": batch[i].text,
                    "source_file": batch[i].source_file,
                    "page_number": batch[i].page_number,
                },
            )
            for i in range(len(batch))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Uploaded {start + len(batch)}/{len(chunks)} chunks")


def search(client: QdrantClient, query: str, top_k: int = 5, score_threshold: float = 0.78):
    """
    Search for the most relevant chunks for a query.

    score_threshold drops results below a minimum relevance score, rather
    than always returning exactly top_k regardless of quality. This exists
    specifically to improve context_precision: without it, retrieval was
    padding out results with loosely-related chunks even when fewer,
    tighter matches existed. 0.78 was chosen by inspecting real query
    scores during testing (strong matches clustered around 0.83-0.88;
    weak/unrelated ones dropped below 0.75) -- treat this as a starting
    point to re-tune if evaluation numbers suggest otherwise.
    """
    query_vector = embed_query(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
    )
    return results.points


if __name__ == "__main__":
    # Full pipeline test: ingest -> clean -> chunk -> embed -> store -> search
    from src.ingestion.pdf_loader import load_all_pdfs
    from src.ingestion.cleaner import strip_repeated_boilerplate
    from src.ingestion.chunker import chunk_pages

    print("Running ingestion pipeline...")
    pages = load_all_pdfs("data")
    pages = strip_repeated_boilerplate(pages)
    chunks, dropped = chunk_pages(pages)
    print(f"{len(chunks)} chunks ready ({dropped} noisy chunks dropped)\n")

    client = get_client()
    create_collection(client, recreate=True)

    print("\nEmbedding and uploading chunks (this will take a few minutes)...")
    upload_chunks(client, chunks)

    print("\n--- Test search ---")
    test_query = "هل هناك حد لإشعار الدائن المرتبط بفاتورة سبق اصدارها؟"
    results = search(client, test_query, top_k=3)
    for r in results:
        print(f"\nscore={r.score:.3f}  page={r.payload['page_number']}  file={r.payload['source_file']}")
        print(r.payload['text'][:200])
        