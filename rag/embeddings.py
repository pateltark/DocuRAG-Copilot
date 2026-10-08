# rag/embeddings.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"⚡ Loading {MODEL_NAME} into RAM...")
        _embedding_model = SentenceTransformer(MODEL_NAME)
        _embedding_model.max_seq_length = 512  # bge-base hard limit
        assert _embedding_model.get_sentence_embedding_dimension() == EMBED_DIM
    return _embedding_model


def embed_docs(chunks: list[str]):
    return get_embedding_model().encode(
        chunks, normalize_embeddings=True, batch_size=16
    )


def embed_query(query: str):
    return get_embedding_model().encode(
        QUERY_PREFIX + query, normalize_embeddings=True
    )