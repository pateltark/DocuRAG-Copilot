import os
import gc
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from groq import Groq

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from rag.embeddings import get_embedding_model
from rag.db import (
    save_emb, related_chunks, load_chat, save_sec_vector, related_sec_chunks,
    update_document_status,
)


load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = get_embedding_model()

# Restrict background execution to 1 file at a time to keep RAM predictable
_ingest_executor = ThreadPoolExecutor(max_workers=1)


def submit_ingest_job(pdf_path: str, user_id: str, document_id: str, filename: str):
    """Queue a document for background ingestion."""
    _ingest_executor.submit(_process_upload, pdf_path, user_id, document_id, filename)


def _process_upload(pdf_path: str, user_id: str, document_id: str, filename: str):
    try:
        create_vectorstore(
            pdf_path=pdf_path,
            user_id=user_id,
            source=filename,
            document_id=document_id,
        )
        update_document_status(document_id, "ready")
    except Exception as e:
        print(f"[ingest] failed for document_id={document_id}: {e}")
        update_document_status(document_id, "failed")
    finally:
        # Force Python memory cleanup after each background job
        gc.collect()


def clean_string(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


def ingest_text(
    text: str,
    user_id: str,
    source: str = None,
    document_id: str = None,
    page_number: int = None,
):
    """Ingest raw string text."""
    text = clean_string(text)
    user_id = clean_string(user_id)
    source = clean_string(source)
    document_id = clean_string(document_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
    )

    docs = [
        Document(
            page_content=text,
            metadata={"source": source} if source else {},
        )
    ]

    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        clean_content = clean_string(chunk.page_content)
        clean_source = clean_string(chunk.metadata.get("source"))

        emb = model.encode(clean_content).tolist()

        try:
            save_emb(
                content=clean_content,
                user_id=user_id,
                embedding=emb,
                source=clean_source,
                document_id=document_id,
                page_number=page_number,
            )
        except Exception as e:
            print("\n===== SAVE_EMB ERROR =====")
            print(type(e).__name__, e)
            raise

    return True


def ingest_sec_text(
    text,
    document_id,
    ticker,
    form_type,
    filename,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    embeddings = model.encode(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        save_sec_vector(
            document_id=document_id,
            ticker=ticker,
            form_type=form_type,
            filename=filename,
            chunk_index=i,
            content=chunk,
            embedding=embedding.tolist(),
        )


def create_vectorstore(
    pdf_path: str,
    user_id: str,
    source: str = None,
    document_id: str = None,
):
    """Extracts text page by page with pypdf, chunks it, embeds it,
    and saves it with the correct page number."""

    clean_user_id = clean_string(user_id)
    clean_source = clean_string(source or pdf_path)
    clean_doc_id = clean_string(document_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
    )

    reader = PdfReader(pdf_path)
    any_content_found = False

    for page_index, page in enumerate(reader.pages):
        try:
            page_text = clean_string(page.extract_text() or "")
        except Exception as e:
            print(f"[ingest] page {page_index + 1} extract failed: {e}")
            continue

        if not page_text or len(page_text.strip()) < 10:
            continue

        real_page_num = page_index + 1  # 1-based page number

        doc = Document(page_content=page_text, metadata={"source": clean_source})
        chunks = splitter.split_documents([doc])

        texts = [clean_string(c.page_content) for c in chunks]
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            continue

        # Batch-encode all chunks of this page in one call
        embeddings = model.encode(texts, batch_size=16)

        for text, emb in zip(texts, embeddings):
            save_emb(
                content=text,
                user_id=clean_user_id,
                embedding=emb.tolist(),
                source=clean_source,
                document_id=clean_doc_id,
                page_number=real_page_num,
            )
            any_content_found = True

    if not any_content_found:
        raise ValueError(
            f"No extractable text in {pdf_path}. It may be a scanned PDF (image-only)."
        )

    return True