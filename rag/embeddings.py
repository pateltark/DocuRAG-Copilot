# rag/embeddings.py
import os

# Set before importing sentence_transformers so the tokenizer setting takes effect
# (prevents Windows thread deadlocks)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 1024

_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Returns a global single instance of SentenceTransformer.
    Loads model weights into RAM only when called for the first time.
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"⚡ Loading {MODEL_NAME} into RAM...")
        _embedding_model = SentenceTransformer(MODEL_NAME)
        _embedding_model.max_seq_length = 1024  # chunks are short; caps memory and time
    return _embedding_model


def embed_docs(chunks: list[str]):
    """Embed document chunks (batch). Returns an array of shape (n, 1024)."""
    return get_embedding_model().encode(
        chunks,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=True,
    )


def embed_query(query: str):
    """Embed a single query. BGE-M3 needs no query prefix."""
    return get_embedding_model().encode(query, normalize_embeddings=True)