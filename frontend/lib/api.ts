// API client for the SEC Edgar Research FastAPI backend.
// The base URL is configurable via NEXT_PUBLIC_API_URL and defaults to the
// local FastAPI dev server.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000"

const TOKEN_KEY = "sec_edgar_token"

export function getToken(): string | null {
  if (typeof window === "undefined") return null
  return window.localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string) {
  if (typeof window === "undefined") return
  window.localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken() {
  if (typeof window === "undefined") return
  window.localStorage.removeItem(TOKEN_KEY)
}

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

type RequestOptions = {
  method?: string
  body?: unknown
  auth?: boolean
  isForm?: boolean
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, auth = true, isForm = false } = options

  const headers: Record<string, string> = {}
  if (auth) {
    const token = getToken()
    if (token) headers["Authorization"] = `Bearer ${token}`
  }

  let payload: BodyInit | undefined
  if (body !== undefined) {
    if (isForm) {
      payload = body as FormData
    } else {
      headers["Content-Type"] = "application/json"
      payload = JSON.stringify(body)
    }
  }

  let res: Response
  try {
    res = await fetch(`${API_URL}${path}`, { method, headers, body: payload })
  } catch {
    throw new ApiError(
      `Could not reach the backend at ${API_URL}. Make sure FastAPI is running and reachable.`,
      0,
    )
  }

  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const data = await res.json()
      if (data?.detail) {
        detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)
      }
    } catch {
      // ignore parse errors
    }
    throw new ApiError(detail, res.status)
  }

  if (res.status === 204) return undefined as T
  const text = await res.text()
  return (text ? JSON.parse(text) : undefined) as T
}

// ---- Types ----
export type ChatMessage = {
  role: "user" | "assistant"
  content: string
}

export type UserDocument = {
  id: string
  filename: string
}

export type SecChatResponse = {
  answer: string
  active_doc: unknown
  active_set: unknown
}

// ---- Auth ----
export function register(input: {
  user_id: string
  email: string
  password: string
  name: string
}) {
  return request<{ message: string }>("/auth/register", {
    method: "POST",
    body: input,
    auth: false,
  })
}

export function login(input: { email: string; password: string }) {
  return request<{ access_token: string; token_type: string }>("/auth/login", {
    method: "POST",
    body: input,
    auth: false,
  })
}

export function health() {
  return request<{ status: string }>("/health", { auth: false })
}

// ---- Documents ----
export function listDocuments() {
  return request<UserDocument[]>("/documents")
}

export function listSecDocuments() {
  return request<unknown[]>("/sec/documents")
}

export function uploadPdf(file: File) {
  const form = new FormData()
  form.append("file", file)
  return request<{ message: string; document_id: string }>("/upload", {
    method: "POST",
    body: form,
    isForm: true,
  })
}

export function deleteDocument(documentId: string) {
  return request<{ message: string }>(`/documents/${documentId}`, {
    method: "DELETE",
  })
}

// ---- Chat ----
export function chatDoc(question: string, documentIds?: string[]) {
  return request<{ answer: string }>("/chat/doc", {
    method: "POST",
    body: { question, document_ids: documentIds ?? null },
  })
}

export function chatSec(question: string) {
  return request<SecChatResponse>("/chat/sec", {
    method: "POST",
    body: { question, document_ids: null },
  })
}

export function secActive() {
  return request<{ active_doc: unknown; active_set: unknown }>("/sec/active")
}

export function getHistory(mode: "sec" | "doc") {
  return request<{ messages: ChatMessage[] }>(`/chat/history?mode=${mode}`)
}

export function clearHistory(mode: "sec" | "doc") {
  return request<{ status: string; message: string }>(`/chat/history?mode=${mode}`, {
    method: "DELETE",
  })
}
