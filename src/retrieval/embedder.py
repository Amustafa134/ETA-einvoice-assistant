"""
Wraps the embedding model.

Uses intfloat/multilingual-e5-base -- runs locally via sentence-transformers,
no API key or cost, and is a strong performer for Arabic retrieval
specifically (part of why it was picked over an English-centric model).

IMPORTANT: e5 models require different prefixes for queries vs documents
("query: " vs "passage: "). Mixing these up or skipping them silently
degrades retrieval quality without throwing any error -- so both are
handled here rather than left as something to remember at call sites.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazily load the model once and reuse it (loading is slow-ish)."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME} (first call only)...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document chunks (for storing in Qdrant)."""
    model = get_model()
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(prefixed, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single search query (for querying Qdrant)."""
    model = get_model()
    embedding = model.encode(f"query: {text}", normalize_embeddings=True)
    return embedding.tolist()


if __name__ == "__main__":
    # Quick sanity check: embed one passage and one query, confirm shapes match.
    passage_vec = embed_passages(["هذا نص تجريبي للتحقق من عمل النموذج"])[0]
    query_vec = embed_query("هل هذا يعمل؟")
    print(f"Passage embedding dim: {len(passage_vec)}")
    print(f"Query embedding dim: {len(query_vec)}")
    assert len(passage_vec) == len(query_vec), "Dimension mismatch!"
    print("OK -- dimensions match.")
    