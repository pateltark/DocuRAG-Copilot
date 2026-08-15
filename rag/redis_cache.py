import os
import json
import hashlib
import numpy as np
import redis
from dotenv import load_dotenv

# Import your unified embedding accessor
from rag.embeddings import get_embedding_model

load_dotenv()

# 1. Initialize Redis Client (protocol=2 fixes 'unknown command HELLO')
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", None),
    db=int(os.getenv("REDIS_DB", 0)),
    decode_responses=True,
    socket_timeout=2.0,
    socket_connect_timeout=2.0,
    protocol=2  # 👈 Added to ensure compatibility with older Redis servers
)


def normalize_text(text: str) -> str:
    """Normalizes query text to maximize exact cache hits."""
    return text.strip().lower().rstrip("?").rstrip(".").rstrip("!")


def build_scope_prefix(mode: str, doc_ids: list | None) -> str:
    """Creates a consistent prefix based on search scope/documents."""
    # Convert every ID to a string to prevent integer join errors
    valid_ids = [str(doc_id) for doc_id in doc_ids] if doc_ids else []
    doc_str = ",".join(sorted(valid_ids)) if valid_ids else "global"
    return f"{mode}:{doc_str}"


# ─────────────────────────────────────────────────────────────────
# Tier 1: Exact Hash Match (~1ms)
# ─────────────────────────────────────────────────────────────────

def generate_exact_key(mode: str, doc_ids: list[str] | None, question: str) -> str:
    scope = build_scope_prefix(mode, doc_ids)
    clean_q = normalize_text(question)
    raw_key = f"{scope}:{clean_q}"
    return f"cache:exact:{hashlib.sha256(raw_key.encode()).hexdigest()}"


def get_exact_cache(mode: str, doc_ids: list[str] | None, question: str) -> dict | None:
    key = generate_exact_key(mode, doc_ids, question)
    try:
        cached_data = redis_client.get(key)
        if cached_data:
            return json.loads(cached_data)
    except Exception as e:
        print(f"⚠️ Tier 1 Read Warning: {e}")
    return None


def set_exact_cache(mode: str, doc_ids: list[str] | None, question: str, response: dict, ttl: int = 86400):
    key = generate_exact_key(mode, doc_ids, question)
    try:
        redis_client.setex(key, ttl, json.dumps(response))
        print(f"✅ Saved Tier 1 (Exact): {key}")
    except Exception as e:
        print(f"❌ Tier 1 Save Error: {e}")


# ─────────────────────────────────────────────────────────────────
# Tier 2: Semantic Vector Similarity Match (~20ms)
# ─────────────────────────────────────────────────────────────────

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a, b = np.array(v1), np.array(v2)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


def get_semantic_cache(
    mode: str, doc_ids: list[str] | None, question: str, threshold: float = 0.92
) -> tuple[dict | None, float]:
    try:
        emb_model = get_embedding_model()
        query_vector = emb_model.encode(question).tolist()
        scope = build_scope_prefix(mode, doc_ids)
        pattern = f"cache:sem:{scope}:*"

        keys = redis_client.keys(pattern)
        best_score = 0.0
        best_payload = None

        for key in keys:
            raw_entry = redis_client.get(key)
            if not raw_entry:
                continue

            entry = json.loads(raw_entry)
            score = cosine_similarity(query_vector, entry["embedding"])

            if score > best_score:
                best_score = score
                best_payload = entry["response"]

        if best_score >= threshold:
            return best_payload, best_score

        return None, best_score
    except Exception as e:
        print(f"⚠️ Tier 2 Read Warning: {e}")
        return None, 0.0


def set_semantic_cache(mode: str, doc_ids: list[str] | None, question: str, response: dict, ttl: int = 86400):
    try:
        # FIXED: Changed model() to get_embedding_model()
        emb_model = get_embedding_model()
        query_vector = emb_model.encode(question).tolist()

        clean_q = normalize_text(question)
        scope = build_scope_prefix(mode, doc_ids)
        hash_id = hashlib.md5(clean_q.encode()).hexdigest()

        key = f"cache:sem:{scope}:{hash_id}"
        payload = {
            "question": question,
            "embedding": query_vector,
            "response": response
        }
        redis_client.setex(key, ttl, json.dumps(payload))
        print(f"✅ Saved Tier 2 (Semantic): {key}")
    except Exception as e:
        print(f"❌ Tier 2 Save Error: {e}")


# ─────────────────────────────────────────────────────────────────
# Unified Pipeline Wrapper
# ─────────────────────────────────────────────────────────────────

def get_cached_response(
    mode: str, doc_ids: list[str] | None, question: str, threshold: float = 0.92
) -> tuple[dict | None, str]:
    """
    Checks Tier 1 first (Exact Hash). If miss, checks Tier 2 (Semantic Similarity).
    Returns (cached_payload, status_string).
    """
    exact_hit = get_exact_cache(mode, doc_ids, question)
    if exact_hit:
        return exact_hit, "EXACT_HIT"

    semantic_hit, score = get_semantic_cache(mode, doc_ids, question, threshold)
    if semantic_hit:
        return semantic_hit, f"SEMANTIC_HIT ({round(score * 100, 1)}%)"

    return None, "MISS"


def save_to_cache(mode: str, doc_ids: list[str] | None, question: str, response: dict, ttl: int = 86400):
    """Saves output to both Tier 1 and Tier 2 caches, ONLY if doc_ids exist and answer is valid."""
    
    # 1. Don't cache if there is no active document scope
    if not doc_ids:
        print(" Cache Skipped: No active document ID provided.")
        return

    # 2. Don't cache fallback/error answers or empty responses
    answer_text = response.get("answer", "") if isinstance(response, dict) else str(response)
    
    ignore_patterns = [
        "please select at least one document",
        "there is no active sec document",
        "does not contain enough information"
    ]
    
    if any(pattern in answer_text.lower() for pattern in ignore_patterns):
        print("Cache Skipped: Fallback or error response detected.")
        return

    # Save to Redis if all checks pass
    set_exact_cache(mode, doc_ids, question, response, ttl)
    set_semantic_cache(mode, doc_ids, question, response, ttl)