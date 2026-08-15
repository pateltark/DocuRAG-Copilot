"use client"

import { useRef, useState } from "react"
import useSWR from "swr"
import { Button } from "@/components/ui/button"
import {
  ApiError,
  deleteDocument,
  listDocuments,
  uploadPdf, // Can be renamed in lib/api later if desired
  type UserDocument,
} from "@/lib/api"
import { FileText, Loader2, Trash2, Upload } from "lucide-react"

const MAX_COMPARE_DOCS = 5

// List of extensions supported by Docling
const ALLOWED_EXTENSIONS = [
  "pdf",
  "docx",
  "doc",
  "pptx",
  "xlsx",
  "csv",
  "txt",
  "md",
  "html",
  "png",
  "jpg",
  "jpeg",
]

const ACCEPT_STRING = ALLOWED_EXTENSIONS.map((ext) => `.${ext}`).join(",")

export function DocumentManager({
  selectedIds,
  onSelectionChange,
}: {
  selectedIds: string[]
  onSelectionChange: (ids: string[]) => void
}) {
  const { data, isLoading, mutate } = useSWR<UserDocument[]>("documents", listDocuments, {
    revalidateOnFocus: false,
  })
  const documents = data ?? []

  const inputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  async function onFileSelected(file: File | undefined) {
    if (!file) return

    const ext = file.name.split(".").pop()?.toLowerCase() || ""
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported file extension (.${ext}). Please upload a supported document or image.`)
      return
    }

    setError(null)
    setUploading(true)
    try {
      await uploadPdf(file)
      await mutate()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed.")
    } finally {
      setUploading(false)
      if (inputRef.current) inputRef.current.value = ""
    }
  }

  async function onDelete(id: string) {
    setDeletingId(id)
    setError(null)
    try {
      await deleteDocument(id)
      onSelectionChange(selectedIds.filter((x) => x !== id))
      await mutate()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Delete failed.")
    } finally {
      setDeletingId(null)
    }
  }

  function toggle(id: string) {
    if (selectedIds.includes(id)) {
      onSelectionChange(selectedIds.filter((x) => x !== id))
    } else {
      if (selectedIds.length >= MAX_COMPARE_DOCS) {
        setError(`You can compare up to ${MAX_COMPARE_DOCS} documents at once.`)
        return
      }
      setError(null)
      onSelectionChange([...selectedIds, id])
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Your Documents
        </h3>
        <span className="text-xs text-muted-foreground">{documents.length}</span>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_STRING}
        className="sr-only"
        onChange={(e) => void onFileSelected(e.target.files?.[0])}
      />
      <Button
        variant="outline"
        size="sm"
        className="w-full"
        onClick={() => inputRef.current?.click()}
        disabled={uploading}
      >
        {uploading ? (
          <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />
        ) : (
          <Upload className="size-3.5" aria-hidden="true" />
        )}
        {uploading ? "Uploading…" : "Upload Document"}
      </Button>

      {error && (
        <p className="rounded-lg bg-destructive/10 px-2.5 py-1.5 text-xs text-destructive" role="alert">
          {error}
        </p>
      )}

      <div className="flex flex-col gap-1">
        {isLoading ? (
          <div className="flex items-center gap-2 px-1 py-2 text-xs text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" aria-hidden="true" /> Loading…
          </div>
        ) : documents.length === 0 ? (
          <p className="px-1 py-2 text-xs text-muted-foreground text-pretty">
            No documents yet. Upload a document to start chatting with it.
          </p>
        ) : (
          documents.map((doc) => {
            const selected = selectedIds.includes(doc.id)
            return (
              <div
                key={doc.id}
                className={
                  "group flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm transition-colors " +
                  (selected ? "bg-primary/10" : "hover:bg-muted")
                }
              >
                <input
                  type="checkbox"
                  checked={selected}
                  onChange={() => toggle(doc.id)}
                  className="size-3.5 shrink-0 accent-primary"
                  aria-label={`Select ${doc.filename} for comparison`}
                />
                <FileText className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
                <span className="flex-1 truncate" title={doc.filename}>
                  {doc.filename}
                </span>
                <button
                  type="button"
                  onClick={() => void onDelete(doc.id)}
                  disabled={deletingId === doc.id}
                  className="shrink-0 text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
                  aria-label={`Delete ${doc.filename}`}
                >
                  {deletingId === doc.id ? (
                    <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />
                  ) : (
                    <Trash2 className="size-3.5" aria-hidden="true" />
                  )}
                </button>
              </div>
            )
          })
        )}
      </div>

      {selectedIds.length > 0 && (
        <p className="text-xs text-muted-foreground">
          {selectedIds.length} selected for comparison.
        </p>
      )}
    </div>
  )
}