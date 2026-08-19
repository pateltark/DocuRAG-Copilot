"use client"

import { useEffect, useRef, useState, type KeyboardEvent } from "react"
import useSWR from "swr"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Button } from "@/components/ui/button"
import {
  ApiError,
  chatDoc,
  chatSec,
  clearHistory,
  getHistory,
  secActive,
  type ActiveDoc,
  type ChatMessage,
} from "@/lib/api"
import { ArrowUp, Eraser, Loader2 } from "lucide-react"

type Mode = "sec" | "doc"

// active_set's shape isn't pinned down yet, so render it defensively —
// never dump a raw object into JSX (that's what crashed last time).
function formatActiveSet(activeSet: unknown): string | null {
  if (activeSet == null) return null
  if (typeof activeSet === "string") return activeSet
  if (Array.isArray(activeSet)) {
    if (activeSet.length === 0) return null
    return `${activeSet.length} filing${activeSet.length > 1 ? "s" : ""}`
  }
  if (typeof activeSet === "object") {
    return "comparing a set"
  }
  return String(activeSet)
}

export function ChatPanel({
  mode,
  selectedDocumentIds,
  chatId,
  onChatIdChange,
}: {
  mode: Mode
  selectedDocumentIds: string[]
  /** null = unsaved "new chat" — no history to load yet */
  chatId: string | null
  /** Called with the real chat_id once the first message lands, and with
   * null after the current chat is cleared/deleted. */
  onChatIdChange: (chatId: string | null) => void
}) {
  const { data, isLoading, mutate } = useSWR(
    ["history", mode, chatId],
    () => (chatId ? getHistory(mode, chatId).then((r) => r.messages) : Promise.resolve([] as ChatMessage[])),
    { revalidateOnFocus: false },
  )

  const [input, setInput] = useState("")
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [clearing, setClearing] = useState(false)
  const [activeDoc, setActiveDoc] = useState<ActiveDoc>(null)
  const [activeSet, setActiveSet] = useState<unknown>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const messages = data ?? []

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [messages, sending])

  // Restore active doc/set on mount and whenever we switch into SEC mode
  // (e.g. after a page refresh), since chat-panel remounts per mode.
  useEffect(() => {
    if (mode !== "sec") return
    let cancelled = false
    secActive()
      .then((res) => {
        if (cancelled) return
        setActiveDoc(res.active_doc ?? null)
        setActiveSet(res.active_set ?? null)
      })
      .catch(() => {
        // non-critical — leave badge empty if this fails
      })
    return () => {
      cancelled = true
    }
  }, [mode])

  async function send() {
    const question = input.trim()
    if (!question || sending) return
    setError(null)
    setSending(true)
    setInput("")

    // Optimistically show the user's message.
    const optimistic: ChatMessage[] = [...messages, { role: "user", content: question }]
    mutate(optimistic, { revalidate: false })

    try {
      if (mode === "sec") {
        const res = await chatSec(question, chatId ?? undefined)
        mutate([...optimistic, { role: "assistant", content: res.answer }], { revalidate: false })
        setActiveDoc(res.active_doc ?? null)
        setActiveSet(res.active_set ?? null)
        onChatIdChange(res.chat_id)
      } else {
        const res = await chatDoc(
          question,
          chatId ?? undefined,
          selectedDocumentIds.length ? selectedDocumentIds : undefined,
        )
        mutate([...optimistic, { role: "assistant", content: res.answer }], { revalidate: false })
        onChatIdChange(res.chat_id)
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to get a response.")
      mutate(messages, { revalidate: false }) // roll back
      setInput(question)
    } finally {
      setSending(false)
    }
  }

  async function onClear() {
    if (clearing) return
    setClearing(true)
    setError(null)
    try {
      if (chatId) {
        await clearHistory(mode, chatId)
      }
      mutate([], { revalidate: false })
      onChatIdChange(null) // back to a fresh "new chat" state
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to clear history.")
    } finally {
      setClearing(false)
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      if (e.nativeEvent.isComposing || e.keyCode === 229) return
      e.preventDefault()
      void send()
    }
  }

  const activeSetLabel = formatActiveSet(activeSet)
  const activeLabel = activeDoc
    ? `${activeDoc.ticker} · ${activeDoc.form_type}`
    : activeSetLabel

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-border px-6 py-4">
        <div>
          <h2 className="text-sm font-semibold">
            {mode === "sec" ? "SEC Filings Chat" : "Document Chat"}
          </h2>
          <p className="text-xs text-muted-foreground">
            {mode === "sec"
              ? "Ask questions across SEC Edgar filings."
              : selectedDocumentIds.length
                ? `Comparing ${selectedDocumentIds.length} selected document${selectedDocumentIds.length > 1 ? "s" : ""}.`
                : "Ask questions about your uploaded documents."}
          </p>
          {mode === "sec" && activeLabel && (
            <div className="mt-1.5 inline-flex items-center gap-1.5 rounded-full border border-border bg-muted px-2.5 py-0.5 text-[11px] text-muted-foreground">
              <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden="true" />
              Active: {activeLabel}
            </div>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClear}
          disabled={clearing || messages.length === 0}
        >
          <Eraser className="size-3.5" aria-hidden="true" />
          Clear
        </Button>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
        {isLoading ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            <Loader2 className="mr-2 size-4 animate-spin" aria-hidden="true" /> Loading history…
          </div>
        ) : messages.length === 0 ? (
          <EmptyState mode={mode} />
        ) : (
          <div className="mx-auto flex max-w-2xl flex-col gap-4">
            {messages.map((m, i) => (
              <MessageBubble key={i} message={m} />
            ))}
            {sending && (
              <div className="flex items-center gap-2 self-start rounded-2xl bg-muted px-4 py-3 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" aria-hidden="true" /> Thinking…
              </div>
            )}
          </div>
        )}
      </div>

      <div className="border-t border-border px-6 py-4">
        <div className="mx-auto max-w-2xl">
          {error && (
            <p className="mb-2 rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive" role="alert">
              {error}
            </p>
          )}
          <div className="flex items-end gap-2 rounded-2xl border border-input bg-background p-2 focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder={
                mode === "sec"
                  ? "Ask about a company's filings…"
                  : "Ask about your documents…"
              }
              className="max-h-40 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none"
            />
            <Button size="icon" onClick={() => void send()} disabled={sending || !input.trim()} aria-label="Send message">
              <ArrowUp className="size-4" aria-hidden="true" />
            </Button>
          </div>
          <p className="mt-2 text-center text-xs text-muted-foreground">
            Press Enter to send, Shift+Enter for a new line.
          </p>
        </div>
      </div>
    </div>
  )
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user"

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-primary px-4 py-3 text-sm text-primary-foreground">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="prose-chat max-w-[85%] rounded-2xl rounded-bl-sm bg-muted px-4 py-3 text-sm text-foreground">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
      </div>
    </div>
  )
}

function EmptyState({ mode }: { mode: Mode }) {
  return (
    <div className="mx-auto flex h-full max-w-md flex-col items-center justify-center text-center">
      <h3 className="text-base font-medium">
        {mode === "sec" ? "Research SEC filings" : "Chat with your documents"}
      </h3>
      <p className="mt-2 text-sm text-muted-foreground text-pretty">
        {mode === "sec"
          ? "Ask about revenue, risk factors, management discussion, or anything in the filings."
          : "Upload PDFs from the sidebar, then ask questions or compare across documents."}
      </p>
    </div>
  )
}