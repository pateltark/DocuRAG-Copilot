import os
import uuid
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import JWTError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import BackgroundTasks
from agent.llm_agent import ask_upload, ask_sec
from rag.emb_chunks import create_vectorstore
from auth import hash_password, verify_password, create_access_token, decode_token
from rag.db import (
    save_user_info, get_user_by_email, get_user_by_id, save_chat, save_emb,
    save_doc_info, update_document_status, get_user_documents, list_sec_documents,
    delete_document, load_chat, clear_chat, list_chats
)
from agent.session import get_active_doc, get_active_set
import asyncio

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


logger = logging.getLogger(__name__)

app = FastAPI(title="SEC Edgar Research API")

security = HTTPBearer()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://65.2.146.75:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    document_ids: list[str] | None = None
    chat_id: str | None = None  # None => start a new chat section

class RegisterRequest(BaseModel):
    user_id: str
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

# ── Updated DocumentOut Pydantic Model ─────────────────────
class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str  # Reflects 'processing', 'ready', or 'failed'


MAX_COMPARE_DOCS = 5
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

ALLOWED_EXTENSIONS = {
    "pdf", "docx", "doc", "pptx", "xlsx", 
    "csv", "txt", "md", "html", "png", "jpg", "jpeg"
}



# ── Rate limiter setup ──────────────────────────────────────
# Reuses the same REDIS_HOST/PORT/PASSWORD/DB env vars as rag/redis_cache.py
def _build_redis_url() -> str:
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    password = os.getenv("REDIS_PASSWORD", "")
    db = os.getenv("REDIS_DB", "0")
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/{db}"


def user_or_ip_key(request: Request) -> str:
    """Rate-limit by authenticated user id when possible, else by IP."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            payload = decode_token(auth_header.removeprefix("Bearer "))
            return f"user:{payload['sub']}"
        except JWTError:
            pass
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=user_or_ip_key, storage_uri=_build_redis_url())
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


RATE_LIMIT_MESSAGES = {
    "/auth/login": "Too many login attempts. Please wait a minute and try again.",
    "/auth/register": "Too many signup attempts. Please wait a minute and try again.",
    "/upload": "Too many uploads. Please wait a minute before uploading again.",
    "/chat/doc": "You're sending messages too fast. Please slow down.",
    "/chat/sec": "You're sending messages too fast. Please slow down.",
}


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    message = RATE_LIMIT_MESSAGES.get(
        request.url.path,
        "You're doing that too often. Please wait a moment and try again."
    )
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limited",
            "message": message,
        },
    )


@app.get("/")
def read_root():
    return {"status": "ok"}


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        token = credentials.credentials
        payload = decode_token(token)
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def _process_document(tmp_path: str, user_id: str, doc_id: str):
    try:
        create_vectorstore(tmp_path, user_id, document_id=doc_id)
        update_document_status(doc_id, "ready")
    except Exception as exc:
        update_document_status(doc_id, "failed")
        logger.error(f"Failed embedding pipeline for document {doc_id}: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(user=Depends(get_current_user)):
    return get_user_documents(user["sub"])


@app.post("/auth/register")
@limiter.limit("5/minute")
def register(request: Request, body: RegisterRequest):
    existing = get_user_by_email(body.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")
    hashed = hash_password(body.password)
    save_user_info(body.user_id, body.email, hashed, body.name)
    return {"message": "User registered successfully."}


@app.post("/auth/login")
@limiter.limit("5/minute")
def login(request: Request, body: LoginRequest):
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["pass_word"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_access_token(user["user_id"], user["email"])
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me")
def get_me(user=Depends(get_current_user)):
    info = get_user_by_id(user["sub"])
    if not info:
        raise HTTPException(status_code=404, detail="User not found.")
    return info


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Refactored /upload Endpoint ─────────────────────────────

@app.post("/upload")
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    filename = file.filename or ""
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '.{ext}'.")

    file_suffix = f".{ext}" if ext else ""
    total = 0
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(1024 * 1024):  # read in 1MB chunks
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit."
                    )
                tmp.write(chunk)
    except HTTPException:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    doc_id = save_doc_info(user["sub"], file.filename, status="processing")
    background_tasks.add_task(_process_document, tmp_path, user["sub"], doc_id)

    return {"message": "Upload received, processing.", "document_id": doc_id, "status": "processing"}

    
# @app.post("/upload")
# async def upload_document(file: UploadFile = File(...), user=Depends(get_current_user)):
#     filename = file.filename or ""
#     ext = filename.split(".")[-1].lower() if "." in filename else ""

#     if ext not in ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file format '.{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
#         )

#     file_suffix = f".{ext}" if ext else ""

#     with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name

#     # 1. Create entry with 'processing' status
#     doc_id = save_doc_info(user["sub"], file.filename, status="processing")

#     try:
#         # 2. Run embedding work inside thread pool
#         await asyncio.to_thread(create_vectorstore, tmp_path, user["sub"], document_id=doc_id)
        
#         # 3. Mark document as 'ready' if processing completes without errors
#         update_document_status(doc_id, "ready")

#     except Exception as exc:
#         # 4. Mark document as 'failed' if any step in extraction/embedding throws an exception
#         update_document_status(doc_id, "failed")
#         logger.error(f"Failed embedding pipeline for document {doc_id}: {exc}")
        
#         raise HTTPException(
#             status_code=500,
#             detail=f"Document processing failed during embedding creation: {str(exc)}"
#         )
        
#     finally:
#         if os.path.exists(tmp_path):
#             os.unlink(tmp_path)

#     return {"message": "Document indexed successfully.", "document_id": doc_id, "status": "ready"}


@app.delete("/upload")
def clear_pdf(user=Depends(get_current_user)):
    pdf_ready = False
    return {"message": "PDF cleared."}


@app.post("/chat/doc")
@limiter.limit("10/minute")
async def chat_with_doc(request: Request, req: QueryRequest, user=Depends(get_current_user)):
    if req.document_ids and len(req.document_ids) > MAX_COMPARE_DOCS:
        raise HTTPException(
            status_code=400,
            detail=f"You can compare up to {MAX_COMPARE_DOCS} documents at once."
        )

    user_id = user["sub"]
    chat_id = req.chat_id or str(uuid.uuid4())

    save_chat(user_id=user_id, role="user", content=req.question, mode="doc", chat_id=chat_id)

    answer = ask_upload(
        question=req.question, 
        user_id=user_id, 
        document_ids=req.document_ids
    )

    save_chat(user_id=user_id, role="assistant", content=answer, mode="doc", chat_id=chat_id)

    return {"answer": answer, "chat_id": chat_id}


@app.post("/chat/sec")
@limiter.limit("10/minute")
async def chat_with_sec(request: Request, req: QueryRequest, user=Depends(get_current_user)):
    user_id = user["sub"]
    chat_id = req.chat_id or str(uuid.uuid4())

    save_chat(user_id=user_id, role="user", content=req.question, mode="sec", chat_id=chat_id)
    answer = ask_sec(question=req.question, user_id=user_id)
    save_chat(user_id=user_id, role="assistant", content=answer, mode="sec", chat_id=chat_id)

    return {
        "answer": answer,
        "chat_id": chat_id,
        "active_doc": get_active_doc(user_id),
        "active_set": get_active_set(user_id),
    } 


@app.get("/sec/active")
def sec_active(user=Depends(get_current_user)):
    user_id = user["sub"]
    return {
        "active_doc": get_active_doc(user_id),
        "active_set": get_active_set(user_id),
    }


@app.get("/sec/documents")
def sec_documents(user=Depends(get_current_user)):
    return list_sec_documents()


@app.delete("/documents/{document_id}")
def delete_document_route(document_id: str, user=Depends(get_current_user)):
    delete_document(user_id=user["sub"], document_id=document_id)
    return {"message": "Document deleted."}


# ── Chat sections (sidebar) ────────────────────────────────
@app.get("/chat/sessions")
def get_chat_sessions(
    mode: str = Query("sec", description="Chat mode: 'sec' or 'doc'"),
    user=Depends(get_current_user),
):
    if mode not in ["sec", "doc"]:
        raise HTTPException(status_code=400, detail="Invalid chat mode")
    return list_chats(user_id=user["sub"], mode=mode)


# ── Chat history (single section) ──────────────────────────
@app.get("/chat/history")
async def get_history(
    mode: str = Query("sec", description="Chat mode: 'sec' or 'doc'"),
    chat_id: str | None = Query(None, description="If omitted, returns all messages for the mode"),
    user=Depends(get_current_user),
):
    user_id = user["sub"]
    if mode not in ["sec", "doc"]:
        raise HTTPException(status_code=400, detail="Invalid chat mode")

    messages = load_chat(user_id=user_id, mode=mode, chat_id=chat_id)
    return {"messages": messages}


@app.delete("/chat/history")
async def clear_history(
    mode: str = Query("sec", description="Chat mode: 'sec' or 'doc'"),
    chat_id: str | None = Query(None, description="If omitted, clears all chats for the mode"),
    user=Depends(get_current_user),
):
    user_id = user["sub"]
    if mode not in ["sec", "doc"]:
        raise HTTPException(status_code=400, detail="Invalid chat mode")

    clear_chat(user_id=user_id, mode=mode, chat_id=chat_id)
    msg = f"Cleared chat {chat_id}" if chat_id else f"Cleared all history for {mode} mode"
    return {"status": "success", "message": msg}