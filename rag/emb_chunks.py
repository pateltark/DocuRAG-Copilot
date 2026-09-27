import os
import gc
import json
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from groq import Groq

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from rag.embeddings import get_embedding_model
from rag.db import (
    save_emb, related_chunks, load_chat, save_sec_vector, related_sec_chunks,
    update_document_status,
)

from pypdf import PdfReader, PdfWriter
import tempfile


load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = get_embedding_model()

# Restrict background execution to 1 file at a time to prevent RAM multiplication
_ingest_executor = ThreadPoolExecutor(max_workers=1)

# ==============================================================================
# 🚀 MEMORY OPTIMIZED DOCLING CONFIGURATION
# ==============================================================================
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = False
pipeline_options.generate_page_images = False   # ADD: stops page rasterization if this flag exists in your version
pipeline_options.images_scale = 1.0   # Disables table AI vision model (saves ~30% RAM & eliminates std::bad_alloc)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


# ==============================================================================


def _split_pdf_into_chunks(pdf_path: str, pages_per_chunk: int = 2):
    """Splits a PDF into smaller temp PDFs. Returns list of
    (chunk_path, start_page_offset) where start_page_offset is the
    0-based index of the first page in this chunk, relative to the
    original document — used to correct page numbers later."""
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    chunks = []

    for start in range(0, total_pages, pages_per_chunk):
        end = min(start + pages_per_chunk, total_pages)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        with open(tmp.name, "wb") as f:
            writer.write(f)

        chunks.append((tmp.name, start))  # start = 0-based offset

    return chunks


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
    pages_per_chunk: int = 2,
):
    """Parses documents with Docling in page-batches to bound native
    memory use, while preserving correct page numbers."""

    clean_user_id = clean_string(user_id)
    clean_source = clean_string(source or pdf_path)
    clean_doc_id = clean_string(document_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
    )

    chunk_paths = _split_pdf_into_chunks(pdf_path, pages_per_chunk=pages_per_chunk)
    any_content_found = False

    try:
        for chunk_path, page_offset in chunk_paths:
            try:
                result = doc_converter.convert(chunk_path)

                pages_map = {}  # local page_num (within chunk) -> list of text snippets

                for item, _ in result.document.iterate_items():
                    local_page_num = None
                    if getattr(item, "prov", None) and len(item.prov) > 0:
                        local_page_num = item.prov[0].page_no

                    snippet = ""
                    if hasattr(item, "export_to_markdown"):
                        try:
                            snippet = item.export_to_markdown(doc=result.document)
                        except Exception:
                            snippet = getattr(item, "text", "")
                    elif hasattr(item, "text") and item.text:
                        snippet = item.text.strip()

                    if snippet and snippet.strip():
                        if local_page_num not in pages_map:
                            pages_map[local_page_num] = []
                        pages_map[local_page_num].append(snippet.strip())

                # Fallback if this chunk yielded nothing structurally
                if not pages_map:
                    full_md = clean_string(result.document.export_to_markdown())
                    if full_md and len(full_md.strip()) >= 50:
                        pages_map = {1: [full_md]}  # local page 1 = only page in a degenerate case

                for local_page_num, snippets in pages_map.items():
                    page_text = clean_string("\n\n".join(snippets))
                    if not page_text or len(page_text.strip()) < 10:
                        continue

                    # Correct page number: local_page_num is 1-based within
                    # the chunk; page_offset is 0-based start of chunk in
                    # the original doc. real_page = offset + local_page_num.
                    real_page_num = (
                        page_offset + local_page_num
                        if local_page_num is not None
                        else None
                    )

                    doc = Document(
                        page_content=page_text,
                        metadata={"source": clean_source}
                    )
                    chunks = splitter.split_documents([doc])

                    for chunk in chunks:
                        clean_content = clean_string(chunk.page_content)
                        if not clean_content or not clean_content.strip():
                            continue

                        emb = model.encode(clean_content).tolist()

                        save_emb(
                            content=clean_content,
                            user_id=clean_user_id,
                            embedding=emb,
                            source=clean_source,
                            document_id=clean_doc_id,
                            page_number=real_page_num,
                        )
                        any_content_found = True

                # Free this chunk's Docling result before moving to the next
                del result
                gc.collect()

            finally:
                # Always remove the temp chunk file, even on failure
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)

        if not any_content_found:
            raise ValueError(f"Docling extracted no usable content from {pdf_path}")

        return True

    finally:
        gc.collect()