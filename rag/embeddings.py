# rag/embeddings.py
import os
from sentence_transformers import SentenceTransformer


"""
    Returns a global single instance of SentenceTransformer.
    Loads model weights from local container disk into RAM when called for the first time.
"""

# Disable parallel tokenizer execution to prevent Windows thread deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_embedding_model = None

def get_embedding_model():
    """
    Returns a global single instance of SentenceTransformer.
    Loads model weights into RAM only when called for the first time.
    """
    global _embedding_model
    if _embedding_model is None:
        print("⚡ Loading SentenceTransformer model into RAM...")
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model