import os
import json
import uuid
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


import os
import psycopg2.pool
from dotenv import load_dotenv

# Load variables from .env file into os.environ
load_dotenv()


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── Connection pool ─────────────────────────────────────────
# Replaces the old single shared `conn`/`cursor` globals. Each request now
# borrows its own connection from the pool and returns it when done, so
# concurrent requests (FastAPI's threadpool, multiple uvicorn workers, or
# multiple ECS/EC2 tasks all pointed at the same DB) don't share a cursor.
#
# All credentials now come from environment variables — set these in your
# docker-compose.yml / ECS task definition / .env (gitignored), never
# hardcoded. If "Login@100" was ever a real password, rotate it — it was
# committed to the public repo.
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=int(os.getenv("DB_POOL_MAX", "10")),
    database=os.getenv("DB_NAME", "ai"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
)

@contextmanager
def get_db():
    """Borrow a connection+cursor from the pool for the duration of one
    call, then return it. Use as: `with get_db() as cursor: ...`"""
    conn = connection_pool.getconn()
    conn.autocommit = True
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
        connection_pool.putconn(conn)


# ── Schema setup (runs once at import, using its own connection) ──────────
def _run_migrations():
    with get_db() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ready',
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_login (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                email VARCHAR(254) NOT NULL,
                pass_word TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sec_documents_table (
                id SERIAL PRIMARY KEY,
                ticker TEXT NOT NULL,
                form_type TEXT NOT NULL,
                filed_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                url TEXT NOT NULL,
                path TEXT NOT NULL,
                downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (ticker, form_type, filed_at)
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sec_vectors (
                id SERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES sec_documents_table(id),
                ticker TEXT NOT NULL,
                form_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,

                embedding VECTOR(384),

                content_tsvector TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(content,''))
                ) STORED
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_emb (
                id SERIAL PRIMARY KEY,

                user_id TEXT NOT NULL,

                content TEXT NOT NULL,

                embedding VECTOR(384),

                content_tsvector TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(content,''))
                ) STORED,

                source TEXT,
                document_id TEXT,
                page_number INTEGER,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Migrations for columns added after initial table creation
        cursor.execute("ALTER TABLE chat_emb ADD COLUMN IF NOT EXISTS document_id TEXT;")
        cursor.execute("ALTER TABLE chat_emb ADD COLUMN IF NOT EXISTS page_number INTEGER;")
        cursor.execute("ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS mode VARCHAR(20) DEFAULT 'sec';")
        cursor.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ready';")
        cursor.execute("""
                    CREATE INDEX IF NOT EXISTS chat_emb_vector_idx
                    ON chat_emb
                    USING hnsw (embedding vector_cosine_ops);
                    """)


        cursor.execute("""
                        CREATE INDEX IF NOT EXISTS sec_vectors_vector_idx
                        ON sec_vectors
                        USING hnsw (embedding vector_cosine_ops);
                        """)


        cursor.execute("""
                    ALTER TABLE chat_emb
                    ADD COLUMN IF NOT EXISTS content_tsvector TSVECTOR
                    GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(content,''))
                    ) STORED;
         
                    """)

        cursor.execute("""
                    ALTER TABLE sec_vectors
                    ADD COLUMN IF NOT EXISTS content_tsvector TSVECTOR
                    GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(content,''))
                    ) STORED;
        """)

        cursor.execute("""
                    CREATE INDEX IF NOT EXISTS chat_emb_user_doc_idx
                    ON chat_emb(user_id, document_id);
                    """)

        cursor.execute("""
                        CREATE INDEX IF NOT EXISTS sec_vectors_doc_idx
                        ON sec_vectors(document_id);
                        """)


_run_migrations()


# ── SEC vectors ─────────────────────────────────────────────
def save_sec_vector(document_id: int, ticker: str, form_type: str, filename: str,
                     chunk_index: int, content: str, embedding):
    with get_db() as cursor:
        cursor.execute(
            """
            INSERT INTO sec_vectors (
                document_id, ticker, form_type, filename, chunk_index, content, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (document_id, ticker, form_type, filename, chunk_index, content, json.dumps(embedding)),
        )


def related_sec_chunks(document_id: int, question: str, k: int = 5, k_rrf: int = 60):
    query_embedding = json.dumps(model.encode(question).tolist())
    
    sql = """
    WITH semantic_search AS (
        SELECT 
            sv.id, sv.content, sv.ticker, sv.form_type, sv.filename,
            ROW_NUMBER() OVER (ORDER BY sv.embedding <=> %s::vector) AS rank,
            (sv.embedding <=> %s::vector) AS distance
        FROM sec_vectors sv
        WHERE sv.document_id = %s
        ORDER BY distance
        LIMIT 20
    ),
    keyword_search AS (
        SELECT 
            sv.id, sv.content, sv.ticker, sv.form_type, sv.filename,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(sv.content_tsvector, plainto_tsquery('english', %s)) DESC
            ) AS rank
        FROM sec_vectors sv
        WHERE sv.document_id = %s 
          AND sv.content_tsvector @@ plainto_tsquery('english', %s)
        ORDER BY ts_rank_cd(sv.content_tsvector, plainto_tsquery('english', %s)) DESC
        LIMIT 20
    )
    SELECT 
        COALESCE(s.content, k.content) AS content,
        COALESCE(s.ticker, k.ticker) AS ticker,
        COALESCE(s.form_type, k.form_type) AS form_type,
        COALESCE(s.filename, k.filename) AS filename,
        COALESCE(s.distance, 1.0) AS distance,
        (
            COALESCE(1.0 / (%s + s.rank), 0.0) +
            COALESCE(1.0 / (%s + k.rank), 0.0)
        ) AS rrf_score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    ORDER BY rrf_score DESC
    LIMIT %s;
    """

    with get_db() as cursor:
        cursor.execute(sql, (
            query_embedding, query_embedding, document_id,
            question, document_id, question, question,
            k_rrf, k_rrf,
            k
        ))
        rows = cursor.fetchall()

    # Returns [(content, ticker, form_type, filename, distance), ...]
    return [(row[0], row[1], row[2], row[3], row[4]) for row in rows]



def get_sec_document(ticker, form_type):
    with get_db() as cursor:
        cursor.execute(
            """
            SELECT id, path
            FROM sec_documents_table
            WHERE ticker = %s AND form_type = %s
            ORDER BY filed_at DESC
            LIMIT 1
            """,
            (ticker, form_type),
        )
        row = cursor.fetchone()
    if not row:
        return None
    return {"id": row[0], "path": row[1]}


def save_sec_document(ticker, form_type, filed_at, filename, url, path):
    with get_db() as cursor:
        cursor.execute(
            """
            INSERT INTO sec_documents_table (ticker, form_type, filed_at, filename, url, path)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, form_type, filed_at) DO NOTHING
            RETURNING id
            """,
            (ticker, form_type, filed_at, filename, url, path),
        )
        row = cursor.fetchone()
        if row:
            return row[0]

        cursor.execute(
            """
            SELECT id FROM sec_documents_table
            WHERE ticker = %s AND form_type = %s AND filed_at = %s
            """,
            (ticker, form_type, filed_at),
        )
        return cursor.fetchone()[0]


def list_sec_documents(limit=50):
    with get_db() as cursor:
        cursor.execute(
            """
            SELECT id, ticker, form_type, filed_at, filename
            FROM sec_documents_table
            ORDER BY filed_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall()
    return [
        {"id": r[0], "ticker": r[1], "form_type": r[2], "filed_at": str(r[3]), "filename": r[4]}
        for r in rows
    ]


# ── Users ───────────────────────────────────────────────────
def save_user_info(user_id, email, pass_word, name):
    with get_db() as cursor:
        cursor.execute(
            """
            INSERT INTO user_login (user_id, email, pass_word, name)
            VALUES (%s, %s, %s, %s)
            """,
            (user_id, email, pass_word, name),
        )


def get_user_by_email(email):
    with get_db() as cursor:
        cursor.execute(
            "SELECT user_id, email, pass_word, name FROM user_login WHERE email = %s",
            (email,),
        )
        row = cursor.fetchone()
    if not row:
        return None
    return {"user_id": row[0], "email": row[1], "pass_word": row[2], "name": row[3]}


# ── Uploaded PDF documents ──────────────────────────────────
def save_doc_info(user_id, pdf_name, status="processing"):
    doc_id = str(uuid.uuid4())
    with get_db() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (id, user_id, filename, status)
            VALUES (%s, %s, %s, %s)
            """,
            (doc_id, user_id, pdf_name, status),
        )
    return doc_id


def update_document_status(document_id, status):
    with get_db() as cursor:
        cursor.execute(
            "UPDATE documents SET status = %s WHERE id = %s",
            (status, document_id),
        )


def count_processing_documents(user_id):
    with get_db() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM documents WHERE user_id = %s AND status = 'processing'",
            (user_id,),
        )
        return cursor.fetchone()[0]


def get_user_documents(user_id):
    with get_db() as cursor:
        cursor.execute(
            """
            SELECT id, filename, status
            FROM documents
            WHERE user_id = %s
            ORDER BY uploaded_at DESC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
    return [{"id": row[0], "filename": row[1], "status": row[2]} for row in rows]


def delete_document(user_id, document_id):
    # Scoped to user_id so a user can only delete their own documents,
    # even if they somehow got hold of another user's document_id.
    with get_db() as cursor:
        cursor.execute(
            "DELETE FROM chat_emb WHERE user_id = %s AND document_id = %s",
            (user_id, document_id),
        )
        cursor.execute(
            "DELETE FROM documents WHERE user_id = %s AND id = %s",
            (user_id, document_id),
        )


# ── PDF RAG embeddings ──────────────────────────────────────
def save_emb(content, user_id, embedding, source=None, document_id=None, page_number=None):
    emb_json = json.dumps(embedding)
    with get_db() as cursor:
        cursor.execute(
            """
            INSERT INTO chat_emb (content, user_id, embedding, source, document_id, page_number)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (content, user_id, emb_json, source, document_id, page_number),
        )


def related_chunks(user_id: str, question: str, k: int = 4, document_ids: list[str] | None = None, k_rrf: int = 60):
    query_embedding = json.dumps(model.encode(question).tolist())
    
    # Dynamic SQL WHERE filters depending on document_ids presence
    doc_filter = "AND document_id = ANY(%s)" if document_ids else ""
    params_semantic = [query_embedding, query_embedding, user_id]
    if document_ids:
        params_semantic.append(document_ids)
        
    params_keyword = [question, user_id]
    if document_ids:
        params_keyword.append(document_ids)

    sql = f"""
    WITH semantic_search AS (
        SELECT 
            id, content, page_number,
            ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank,
            (embedding <=> %s::vector) AS distance
        FROM chat_emb
        WHERE user_id = %s {doc_filter}
        ORDER BY distance
        LIMIT 20
    ),
    keyword_search AS (
        SELECT 
            id, content, page_number,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC) AS rank
        FROM chat_emb, plainto_tsquery('english', %s) query
        WHERE user_id = %s {doc_filter}
          AND to_tsvector('english', content) @@ query
        ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC
        LIMIT 20
    )
    SELECT 
        COALESCE(s.content, k.content) AS content,
        COALESCE(s.page_number, k.page_number) AS page_number,
        COALESCE(s.distance, 1.0) AS distance,
        (
            COALESCE(1.0 / (%s + s.rank), 0.0) +
            COALESCE(1.0 / (%s + k.rank), 0.0)
        ) AS rrf_score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    ORDER BY rrf_score DESC
    LIMIT %s;
    """

    all_params = params_semantic + params_keyword + [k_rrf, k_rrf, k]

    with get_db() as cursor:
        cursor.execute(sql, all_params)
        rows = cursor.fetchall()

    # Returns [(content, page_number, distance), ...] matching existing format
    return [(row[0], row[1], row[2]) for row in rows]


def related_chunks_per_doc(
    user_id: str, 
    question: str, 
    document_ids: list[str], 
    k_per_doc: int = 10,   # 1. Increased default candidate count from 4 to 10
    k_rrf: int = 60
):
    query_embedding = json.dumps(model.encode(question).tolist())
    
    # 2. Scale CTE limit so RRF always has enough depth to fuse semantic + keyword results
    cte_limit = max(30, k_per_doc * 3)
    
    results = []
    
    sql = """
    WITH semantic_search AS (
        SELECT 
            ce.id,
            ce.content, 
            ce.document_id, 
            d.filename, 
            ce.page_number, 
            ROW_NUMBER() OVER (ORDER BY ce.embedding <=> %s::vector) AS rank,
            (ce.embedding <=> %s::vector) AS distance
        FROM chat_emb ce
        LEFT JOIN documents d ON d.id = ce.document_id
        WHERE ce.user_id = %s AND ce.document_id = %s
        ORDER BY distance
        LIMIT %s
    ),
    keyword_search AS (
        SELECT 
            ce.id,
            ce.content, 
            ce.document_id, 
            d.filename, 
            ce.page_number, 
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(ce.content_tsvector, query) DESC) AS rank
        FROM chat_emb ce
        LEFT JOIN documents d ON d.id = ce.document_id,
        plainto_tsquery('english', %s) query
        WHERE ce.user_id = %s AND ce.document_id = %s
          AND ce.content_tsvector @@ query
        ORDER BY ts_rank_cd(ce.content_tsvector, query) DESC
        LIMIT %s
    )
    SELECT 
        COALESCE(s.content, k.content) AS content,
        COALESCE(s.document_id, k.document_id) AS document_id,
        COALESCE(s.filename, k.filename, 'Uploaded Document') AS filename,
        COALESCE(s.page_number, k.page_number) AS page_number,
        COALESCE(s.distance, 1.0) AS distance,
        (
            COALESCE(1.0 / (%s + s.rank), 0.0) +
            COALESCE(1.0 / (%s + k.rank), 0.0)
        ) AS rrf_score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    ORDER BY rrf_score DESC
    LIMIT %s;
    """

    with get_db() as cursor:
        for doc_id in document_ids:
            cursor.execute(
                sql,
                (
                    query_embedding, query_embedding, user_id, doc_id, cte_limit,  # Semantic params
                    question, user_id, doc_id, cte_limit,                         # Keyword params
                    k_rrf, k_rrf,                                                 # RRF constant params
                    k_per_doc                                                     # Final return limit per doc
                )
            )
            results.append(cursor.fetchall())
            
    return results


# ── Chat history ────────────────────────────────────────────
def save_chat(user_id, role, content, mode='sec'):
    with get_db() as cursor:
        cursor.execute(
            """
            INSERT INTO chat_history (user_id, role, content, mode)
            VALUES (%s, %s, %s, %s)
            """,
            (user_id, role, content, mode),
        )


def load_chat(user_id, mode=None):
    with get_db() as cursor:
        if mode:
            cursor.execute(
                """
                SELECT role, content
                FROM chat_history
                WHERE user_id = %s AND mode = %s
                ORDER BY id
                """,
                (user_id, mode),
            )
        else:
            cursor.execute(
                """
                SELECT role, content
                FROM chat_history
                WHERE user_id = %s
                ORDER BY id
                """,
                (user_id,),
            )
        rows = cursor.fetchall()
    return [{"role": "user" if role == "user" else "assistant", "content": content} for role, content in rows]


def clear_chat(user_id: str, mode: str = None):
    with get_db() as cursor:
        if mode:
            cursor.execute(
                "DELETE FROM chat_history WHERE user_id = %s AND mode = %s",
                (user_id, mode),
            )
        else:
            cursor.execute(
                "DELETE FROM chat_history WHERE user_id = %s",
                (user_id,),
            )