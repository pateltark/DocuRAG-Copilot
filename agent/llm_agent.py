import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import os
from collections import defaultdict
from dotenv import load_dotenv
from groq import Groq

from rag.db import (
    load_chat, save_chat, get_sec_document, save_sec_document,
    related_sec_chunks, save_sec_vector, related_chunks,
    save_doc_info, related_chunks_per_doc
)
from agent.tools import TOOLS
from agent.executor import run_tool
from agent.session import set_active_doc, get_active_doc, set_active_set
from rag.redis_cache import get_cached_response, save_to_cache

from sec.edgar import fetch_sec_filings, download_doc, extract_text
from agent.gaurd_before_fetch import check_avail_sec_doc, planner
from rag.emb_chunks import ingest_text, ingest_sec_text, create_vectorstore
from rag.retrieve import get_retrieve

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Lazy-Loaded Reranker ───────────────────────────────────────────
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("⚡ Loading BAAI/bge-reranker-large...")
        _reranker = CrossEncoder("BAAI/bge-reranker-large")
    return _reranker


def rerank_chunks(query: str, chunks: list, top_k: int = 4) -> list:
    if not chunks:
        return []

    flat_chunks = []
    if isinstance(chunks, list) and len(chunks) > 0 and isinstance(chunks[0], list):
        for group in chunks:
            flat_chunks.extend(group)
    else:
        flat_chunks = chunks

    if not flat_chunks:
        return []

    pairs = [[query, chunk[0]] for chunk in flat_chunks]
    reranker = get_reranker()
    scores = reranker.predict(pairs)

    scored_chunks = list(zip(scores, flat_chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return [chunk for score, chunk in scored_chunks[:top_k]]


RELEVANCE_THRESHOLD = 1.0
DEBUG_RELEVANCE = os.getenv("DEBUG_RELEVANCE") == "1"


def format_citations(context_chunks, mode: str = "doc") -> str:
    if not context_chunks or isinstance(context_chunks, str):
        return ""

    if mode == "sec":
        sec_sources = set()
        flat_sec_rows = []
        if isinstance(context_chunks, list) and len(context_chunks) > 0 and isinstance(context_chunks[0], list):
            for group in context_chunks:
                flat_sec_rows.extend(group)
        else:
            flat_sec_rows = context_chunks

        for row in flat_sec_rows:
            if len(row) >= 4:
                ticker = row[1] if row[1] else "SEC"
                form_type = row[2] if row[2] else "Filing"
                filename = row[3] if row[3] else "document.htm"
                sec_sources.add(f"**{ticker} {form_type}** ({filename})")
            elif len(row) == 3:
                filename = row[1] if row[1] else "SEC Filing"
                sec_sources.add(f"**{filename}**")

        if sec_sources:
            return "\n\n**Sources:** " + " | ".join(sorted(list(sec_sources)))
        return ""

    doc_pages = defaultdict(set)
    flat_pdf_rows = []
    if isinstance(context_chunks, list) and len(context_chunks) > 0 and isinstance(context_chunks[0], list):
        for group in context_chunks:
            flat_pdf_rows.extend(group)
    else:
        flat_pdf_rows = context_chunks

    for row in flat_pdf_rows:
        filename = "Document"
        page_num = None
        if len(row) == 4:
            filename = row[1] if row[1] else "Document"
            page_num = row[2]
        elif len(row) >= 5:
            filename = row[2] if row[2] else "Document"
            page_num = row[3]

        if page_num is not None:
            try:
                doc_pages[filename].add(int(page_num))
            except (ValueError, TypeError):
                doc_pages[filename].add(None)
        else:
            doc_pages[filename].add(None)

    if not doc_pages:
        return ""

    citation_parts = []
    for doc_name, pages in doc_pages.items():
        valid_pages = sorted([p for p in pages if p is not None])
        if valid_pages:
            page_str = ", ".join(str(p) for p in valid_pages)
            label = "Page" if len(valid_pages) == 1 else "Pages"
            citation_parts.append(f"**{doc_name}** ({label}: {page_str})")
        else:
            citation_parts.append(f"**{doc_name}**")

    return "\n\n**Sources:** " + " | ".join(citation_parts)


def contextualize_question(question: str, user_id: str, mode: str = "sec") -> str:
    history = load_chat(user_id, mode=mode)[-4:]
    if not history:
        return question

    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    rewrite_prompt = f"""Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone search query containing all context. Do NOT answer the question.

Chat History:
{history_text}

Follow-up Question: {question}

Standalone Search Query:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Query Rewriter Error] {e}")
        return question


def _best_distance(chunks):
    if not chunks:
        return float("inf")
    if isinstance(chunks[0], list):
        distances = [row[-1] for group in chunks for row in group]
    else:
        distances = [row[-1] for row in chunks]
    return min(distances) if distances else float("inf")


def _is_relevant(chunks, threshold: float = RELEVANCE_THRESHOLD) -> bool:
    distance = _best_distance(chunks)
    return distance <= threshold


def generate_answer(
    question: str,
    context_chunks,
    user_id: str,
    mode: str = "doc",
    labels: list[str] | None = None
) -> str:
    NOT_ENOUGH_INFO = "The provided document context does not contain enough information to answer this question."

    if not context_chunks:
        return NOT_ENOUGH_INFO

    flat_rows = []
    if isinstance(context_chunks, list) and len(context_chunks) > 0 and isinstance(context_chunks[0], list):
        for group in context_chunks:
            flat_rows.extend(group)
    else:
        flat_rows = context_chunks

    if not flat_rows:
        return NOT_ENOUGH_INFO

    context_blocks = []
    for idx, row in enumerate(flat_rows, start=1):
        content = row[0]
        if mode == "sec":
            ticker = row[1] if len(row) > 1 and row[1] else "SEC"
            form_type = row[2] if len(row) > 2 and row[2] else "Filing"
            filename = row[3] if len(row) > 3 and row[3] else ""
            header = f"[Source {idx}: {ticker} {form_type} - {filename}]"
        else:
            if len(row) == 4:
                fname = row[1] or "Document"
                pnum = f", Page {row[2]}" if row[2] is not None else ""
            elif len(row) >= 5:
                fname = row[2] or "Document"
                pnum = f", Page {row[3]}" if row[3] is not None else ""
            else:
                fname = "Document"
                pnum = ""
            header = f"[Source {idx}: {fname}{pnum}]"

        context_blocks.append(f"{header}\n{content}")

    formatted_context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an expert research copilot. Answer the user's question accurately "
        "and concisely using STRICTLY the provided document context below.\n\n"
        "Rules:\n"
        "1. Base your answer ONLY on the provided context.\n"
        "2. If the context does not contain enough information, state EXACTLY:\n"
        f"   '{NOT_ENOUGH_INFO}'\n"
        "3. Do NOT make up citations; sources will be automatically appended."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion: {question}"}
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[generate_answer Error]: {e}")
        return "An error occurred while generating the response."

    if NOT_ENOUGH_INFO.lower() in answer_text.lower():
        return NOT_ENOUGH_INFO

    citations = format_citations(context_chunks, mode=mode)
    return f"{answer_text}{citations}"


def ask_sec(question: str, user_id: str):
    NOT_ENOUGH_INFO = "The provided document context does not contain enough information to answer this question."

    plan = planner(question)
    search_query = question

    # ── CASE 1: Querying Currently Active SEC Document ───────────────
    if plan.action == "CURRENT_DOC":
        active_doc = get_active_doc(user_id)
        if not active_doc:
            return "There is no active SEC document. Please ask for a filing first (e.g., 'Show Tesla's latest 10-K')."

        doc_id = active_doc["document_id"]
        doc_ids = [doc_id]  # 👈 Dynamic doc_id scope

        # 1. Check cache WITH doc_ids
        cached_payload, cache_status = get_cached_response(mode="sec", doc_ids=doc_ids, question=question)
        if cached_payload:
            return cached_payload.get("answer", cached_payload) if isinstance(cached_payload, dict) else cached_payload

        chunks = related_sec_chunks(document_id=doc_id, question=question, k=15)

        if not chunks or not _is_relevant(chunks):
            search_query = contextualize_question(question, user_id)
            chunks = related_sec_chunks(document_id=doc_id, question=search_query, k=15)

        if not chunks:
            return NOT_ENOUGH_INFO

        reranked_chunks = rerank_chunks(query=search_query, chunks=chunks, top_k=4)
        if not reranked_chunks:
            return NOT_ENOUGH_INFO

        sec_crnt_ans = generate_answer(search_query, reranked_chunks, user_id, mode="sec")
        
        # 2. Save cache WITH doc_ids
        save_to_cache(mode="sec", doc_ids=doc_ids, question=question, response={"answer": sec_crnt_ans}, ttl=86400)
        return sec_crnt_ans

    # ── CASE 2: Fetching / Searching SEC Filings ─────────────────────
    docs = check_avail_sec_doc(plan)
    sec_chunks, touched_docs = [], []

    for doc in docs:
        request, db_row = doc["request"], doc["db_row"]

        if db_row:
            document_id = db_row["id"]
            set_active_doc(user_id=user_id, document_id=document_id, ticker=request.ticker, form_type=request.form_type)
        else:
            filings = fetch_sec_filings(ticker=request.ticker, form_type=request.form_type)
            if not filings:
                continue
            filing = filings[0]
            path = download_doc(filing["url"], filing["filename"])
            text = extract_text(path)

            document_id = save_sec_document(
                ticker=request.ticker, form_type=request.form_type,
                filed_at=filing["filed_at"], filename=filing["filename"],
                url=filing["url"], path=path
            )
            ingest_sec_text(text=text, document_id=document_id, ticker=request.ticker, form_type=request.form_type, filename=filing["filename"])
            set_active_doc(user_id=user_id, document_id=document_id, ticker=request.ticker, form_type=request.form_type)

        touched_docs.append({"document_id": document_id, "ticker": request.ticker, "form_type": request.form_type})

    # Collect all document IDs touched by this query
    doc_ids = [str(d["document_id"]) for d in touched_docs]

    # Check cache WITH the specific doc_ids
    if doc_ids:
        cached_payload, cache_status = get_cached_response(mode="sec", doc_ids=doc_ids, question=question)
        if cached_payload:
            return cached_payload.get("answer", cached_payload) if isinstance(cached_payload, dict) else cached_payload

    # Perform retrieval across touched documents
    for t_doc in touched_docs:
        doc_chunks = related_sec_chunks(document_id=t_doc["document_id"], question=question, k=15)
        if doc_chunks:
            sec_chunks.extend(doc_chunks)

    if not sec_chunks or not _is_relevant(sec_chunks):
        search_query = contextualize_question(question, user_id)
        sec_chunks = []
        for t_doc in touched_docs:
            doc_chunks = related_sec_chunks(document_id=t_doc["document_id"], question=search_query, k=15)
            if doc_chunks:
                sec_chunks.extend(doc_chunks)

    if not sec_chunks:
        return NOT_ENOUGH_INFO

    reranked_chunks = rerank_chunks(query=search_query, chunks=sec_chunks, top_k=4)
    if not reranked_chunks:
        return NOT_ENOUGH_INFO

    set_active_set(user_id, touched_docs)
    sec_ans = generate_answer(search_query, reranked_chunks, user_id, mode="sec")
    
    # Save cache WITH specific doc_ids
    save_to_cache(mode="sec", doc_ids=doc_ids, question=question, response={"answer": sec_ans}, ttl=86400)
    return sec_ans


def ask_upload(question: str, user_id: str, document_ids: list[str] | None = None):
    NOT_ENOUGH_INFO = "The provided document context does not contain enough information to answer this question."

    if not document_ids:
        return "Please select at least one document to chat with."

    # 1. Check cache WITH selected document_ids
    cached_payload, cache_status = get_cached_response(mode="doc", doc_ids=document_ids, question=question)
    if cached_payload:
        return cached_payload.get("answer", cached_payload) if isinstance(cached_payload, dict) else cached_payload

    # 2. Fetch context chunks from specific documents
    raw_chunks = related_chunks_per_doc(
        user_id=user_id, question=question, document_ids=document_ids, k_per_doc=10
    )
    raw_chunks = [group for group in raw_chunks if group]

    if not raw_chunks:
        return NOT_ENOUGH_INFO

    reranked_chunks = rerank_chunks(query=question, chunks=raw_chunks, top_k=4)
    if not reranked_chunks:
        return NOT_ENOUGH_INFO

    labels = list({row[2] for row in reranked_chunks if len(row) > 2 and row[2]})
    ans = generate_answer(question=question, context_chunks=reranked_chunks, user_id=user_id, mode="doc", labels=labels)
    
    # 3. Save cache WITH selected document_ids
    save_to_cache(mode="doc", doc_ids=document_ids, question=question, response={"answer": ans}, ttl=86400)
    return ans