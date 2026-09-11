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

# NOTE: every related_*_chunks() query in rag/db.py returns rrf_score as the
# LAST column, not raw cosine distance. rrf_score = 1/(k_rrf+rank_semantic) +
# 1/(k_rrf+rank_keyword) is a "HIGHER is better" score, capped at roughly
# 2/(k_rrf+1) ≈ 0.033 when k_rrf=60. It is NOT a distance in [0, 2], so a
# threshold of 1.45 never rejects anything. Threshold below is a MINIMUM
# acceptable rrf_score, not a maximum distance. Tune empirically — start low
# and raise it if truly irrelevant chunks still get treated as "relevant".
RELEVANCE_THRESHOLD = 0.01
DEBUG_RELEVANCE = os.getenv("DEBUG_RELEVANCE") == "1"

TOP_K = 4


def _flatten(chunks):
    """Flatten a possibly-grouped list of chunk rows into a single list."""
    if not chunks:
        return []
    if isinstance(chunks, list) and len(chunks) > 0 and isinstance(chunks[0], list):
        flat = []
        for group in chunks:
            flat.extend(group)
        return flat
    return chunks


def select_top_chunks(chunks, top_k: int = TOP_K) -> list:
    """
    Flattens grouped chunks, sorts by rrf_score DESCENDING (higher score =
    more relevant — this is a rank-fusion score, not a distance), and
    returns the top_k best-ranked chunks.

    NOTE: this is a purely global sort. When `chunks` contains groups
    from MULTIPLE documents, a global top_k can starve out documents
    entirely (all top_k slots taken by one document's chunks). Use
    select_top_chunks_per_group() instead for multi-document retrieval.
    """
    flat_chunks = _flatten(chunks)
    if not flat_chunks:
        return []

    try:
        # FIX: was ascending (treated rrf_score as a distance). rrf_score is
        # "higher is better", so the best chunks must sort first with
        # reverse=True — otherwise the worst-ranked chunks get selected.
        flat_chunks = sorted(flat_chunks, key=lambda row: row[-1], reverse=True)
    except (TypeError, IndexError):
        # If rows don't have a sortable score field, fall back to
        # original order rather than failing.
        pass

    return flat_chunks[:top_k]


def select_top_chunks_per_group(grouped_chunks, per_group_k: int = 3, overall_cap: int = 16) -> list:
    """
    Diversity-aware chunk selection for multi-document retrieval.

    Unlike select_top_chunks() (a pure global sort), this guarantees every
    group (i.e. every document) contributes up to `per_group_k` of its own
    best chunks, sorted by rrf_score within that group (higher = better).
    This prevents one document's chunks from crowding out every other
    selected document when answering comparison/cross-document questions.

    `grouped_chunks` is expected to be a list of groups, each group being
    a list of chunk rows for one document (the shape returned by
    related_chunks_per_doc). A flat (non-grouped) list is treated as a
    single group.

    The combined result across all groups is then capped at `overall_cap`
    (sorted by rrf_score, best first) so very large multi-doc selections
    still stay within a reasonable size for the LLM context window.
    """
    if not grouped_chunks:
        return []

    if not isinstance(grouped_chunks[0], list):
        grouped_chunks = [grouped_chunks]

    selected = []
    for group in grouped_chunks:
        if not group:
            continue
        try:
            # FIX: descending — rrf_score is "higher is better".
            group_sorted = sorted(group, key=lambda row: row[-1], reverse=True)
        except (TypeError, IndexError):
            group_sorted = group
        selected.extend(group_sorted[:per_group_k])

    if not selected:
        return []

    try:
        # FIX: descending here too, for the same reason.
        selected = sorted(selected, key=lambda row: row[-1], reverse=True)
    except (TypeError, IndexError):
        pass

    return selected[:overall_cap]


def format_citations(context_chunks, mode: str = "doc") -> str:
    if not context_chunks or isinstance(context_chunks, str):
        return ""

    if mode == "sec":
        sec_sources = set()
        flat_sec_rows = _flatten(context_chunks)

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
    flat_pdf_rows = _flatten(context_chunks)

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
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0,
            timeout=30,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Query Rewriter Error] {e}")
        return question


def _best_score(chunks):
    """Highest rrf_score across all candidate chunks. Renamed from
    _best_distance — this is a rank-fusion score (higher = more relevant),
    not a distance, so we take max(), not min()."""
    if not chunks:
        return 0.0
    if isinstance(chunks[0], list):
        scores = [row[-1] for group in chunks for row in group]
    else:
        scores = [row[-1] for row in chunks]
    return max(scores) if scores else 0.0


def _is_relevant(chunks, threshold: float = RELEVANCE_THRESHOLD) -> bool:
    """FIX: score must be >= threshold now (was: distance <= threshold,
    which is backwards for an rrf_score and was always True)."""
    score = _best_score(chunks)
    return score >= threshold


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

    flat_rows = _flatten(context_chunks)

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
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion: {question}"}
            ],
            temperature=0.1,
            max_tokens=1024,
            timeout=30,
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

        top_chunks = select_top_chunks(chunks, top_k=TOP_K)
        if not top_chunks:
            return NOT_ENOUGH_INFO

        sec_crnt_ans = generate_answer(search_query, top_chunks, user_id, mode="sec")

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

    top_chunks = select_top_chunks(sec_chunks, top_k=TOP_K)
    if not top_chunks:
        return NOT_ENOUGH_INFO

    set_active_set(user_id, touched_docs)
    sec_ans = generate_answer(search_query, top_chunks, user_id, mode="sec")

    # Save cache WITH specific doc_ids
    save_to_cache(mode="sec", doc_ids=doc_ids, question=question, response={"answer": sec_ans}, ttl=86400)
    return sec_ans


def ask_upload(question: str, user_id: str, document_ids: list[str] | None = None):
    NOT_ENOUGH_INFO = "The provided document context does not contain enough information to answer this question."

    if not document_ids:
        return "Please select at least one document to chat with."

    cached_payload, cache_status = get_cached_response(mode="doc", doc_ids=document_ids, question=question)
    if cached_payload:
        return cached_payload.get("answer", cached_payload) if isinstance(cached_payload, dict) else cached_payload

    search_query = question
    is_multi_doc = len(document_ids) > 1

    raw_chunks = related_chunks_per_doc(
        user_id=user_id, question=question, document_ids=document_ids, k_per_doc=10
    )
    raw_chunks = [group for group in raw_chunks if group]

    if DEBUG_RELEVANCE:
        print(f"[DEBUG] raw_chunks groups: {len(raw_chunks)}, total rows: {sum(len(g) for g in raw_chunks)}")
        print(f"[DEBUG] best rrf_score: {_best_score(raw_chunks)}, threshold: {RELEVANCE_THRESHOLD}")
        for group in raw_chunks:
            for row in sorted(group, key=lambda r: r[-1], reverse=True)[:3]:
                print(f"[DEBUG CONTENT] score={row[-1]:.4f} {row[0][:200]!r}")

    if not raw_chunks or (not is_multi_doc and not _is_relevant(raw_chunks)):
        search_query = contextualize_question(question, user_id, mode="doc")
        if DEBUG_RELEVANCE:
            print(f"[DEBUG] rewritten query: {search_query}")
        reworded_chunks = related_chunks_per_doc(
            user_id=user_id, question=search_query, document_ids=document_ids, k_per_doc=10
        )
        reworded_chunks = [group for group in reworded_chunks if group]
        if DEBUG_RELEVANCE:
            print(f"[DEBUG] reworded_chunks groups: {len(reworded_chunks)}")
        if reworded_chunks:
            raw_chunks = reworded_chunks

    if not raw_chunks:
        if DEBUG_RELEVANCE:
            print("[DEBUG] EXIT: raw_chunks empty after retry -> NOT_ENOUGH_INFO")
        return NOT_ENOUGH_INFO

    if is_multi_doc:
        top_chunks = select_top_chunks_per_group(raw_chunks, per_group_k=4, overall_cap=20)
    else:
        top_chunks = select_top_chunks(raw_chunks, top_k=TOP_K)

    if DEBUG_RELEVANCE:
        print(f"[DEBUG] top_chunks selected: {len(top_chunks)}")

    if not top_chunks:
        if DEBUG_RELEVANCE:
            print("[DEBUG] EXIT: top_chunks empty -> NOT_ENOUGH_INFO")
        return NOT_ENOUGH_INFO

    labels = list({row[2] for row in top_chunks if len(row) > 2 and row[2]})
    ans = generate_answer(question=search_query, context_chunks=top_chunks, user_id=user_id, mode="doc", labels=labels)
    if DEBUG_RELEVANCE:
        print(f"[DEBUG] final answer starts with: {ans[:80]}")

    save_to_cache(mode="doc", doc_ids=document_ids, question=question, response={"answer": ans}, ttl=86400)
    return ans