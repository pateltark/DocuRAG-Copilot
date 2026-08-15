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

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = get_embedding_model()

# Restrict background execution to 1 file at a time to prevent RAM multiplication
_ingest_executor = ThreadPoolExecutor(max_workers=1)

# ==============================================================================
# 🚀 MEMORY OPTIMIZED DOCLING CONFIGURATION
# ==============================================================================
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False             # Disables image OCR (saves ~60% RAM)
pipeline_options.do_table_structure = False  # Disables table AI vision model (saves ~30% RAM & eliminates std::bad_alloc)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
# ==============================================================================


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
    """Parses documents with Docling while preserving page numbers and saving memory."""
    
    try:
        # 1. Convert document using memory-optimized Docling pipeline
        result = doc_converter.convert(pdf_path)

        clean_user_id = clean_string(user_id)
        clean_source = clean_string(source or pdf_path)
        clean_doc_id = clean_string(document_id)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=60,
        )

        # 2. Group document text page-by-page safely
        pages_map = {}  # page_number -> list of text snippets

        for item, _ in result.document.iterate_items():
            page_num = None
            if getattr(item, "prov", None) and len(item.prov) > 0:
                page_num = item.prov[0].page_no

            snippet = ""
            if hasattr(item, "export_to_markdown"):
                try:
                    # Pass root document reference safely
                    snippet = item.export_to_markdown(doc=result.document)
                except Exception:
                    snippet = getattr(item, "text", "")
            elif hasattr(item, "text") and item.text:
                snippet = item.text.strip()

            if snippet and snippet.strip():
                if page_num not in pages_map:
                    pages_map[page_num] = []
                pages_map[page_num].append(snippet.strip())

        # Fallback if structural iteration yields nothing
        if not pages_map:
            full_md = clean_string(result.document.export_to_markdown())
            if not full_md or len(full_md.strip()) < 50:
                raise ValueError(f"Docling extracted no usable content from {pdf_path}")
            pages_map = {None: [full_md]}

        # 3. Chunk page-by-page so vector store retains exact page numbers
        for page_num, snippets in pages_map.items():
            page_text = clean_string("\n\n".join(snippets))
            if not page_text or len(page_text.strip()) < 10:
                continue

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
                    page_number=page_num,  # 1-based page numbers preserved for PDFs!
                )

        return True

    finally:
        # Run Garbage Collector immediately to free up C++ memory allocations
        gc.collect()