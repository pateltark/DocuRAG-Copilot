"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { ChatPanel } from "@/components/chat-panel"
import { DocumentManager } from "@/components/document-manager"
import { useAuth } from "@/lib/auth-context"
import { FileSearch, FileText, Landmark, LogOut } from "lucide-react"

type Mode = "sec" | "doc"

export function Workspace() {
  const { logout } = useAuth()
  const [mode, setMode] = useState<Mode>("sec")
  const [selectedIds, setSelectedIds] = useState<string[]>([])

  return (
    <div className="flex h-svh bg-background">
      <aside className="flex w-72 shrink-0 flex-col border-r border-border bg-card">
        <div className="flex items-center gap-2.5 border-b border-border px-4 py-4">
          <div className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <FileSearch className="size-4" aria-hidden="true" />
          </div>
          <span className="text-sm font-semibold">DocuRAG Copilot</span>
        </div>

        <nav className="flex flex-col gap-1 p-3">
          <ModeButton
            active={mode === "sec"}
            onClick={() => setMode("sec")}
            icon={<Landmark className="size-4" aria-hidden="true" />}
            label="SEC Filings Chat"
          />
          <ModeButton
            active={mode === "doc"}
            onClick={() => setMode("doc")}
            icon={<FileText className="size-4" aria-hidden="true" />}
            label="Document Chat"
          />
        </nav>

        <div className="flex-1 overflow-y-auto border-t border-border p-4">
          {mode === "doc" ? (
            <DocumentManager selectedIds={selectedIds} onSelectionChange={setSelectedIds} />
          ) : (
            <div className="text-xs text-muted-foreground text-pretty">
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide">About</h3>
              <p>
                Ask questions across SEC Edgar filings. Switch to Document Chat to upload and
                query your own PDFs.
              </p>
            </div>
          )}
        </div>

        <div className="border-t border-border p-3">
          <Button variant="ghost" size="sm" className="w-full justify-start" onClick={logout}>
            <LogOut className="size-4" aria-hidden="true" />
            Sign out
          </Button>
        </div>
      </aside>

      <main className="flex-1 overflow-hidden">
        {/* Keying by mode remounts the panel so history reloads per mode. */}
        <ChatPanel key={mode} mode={mode} selectedDocumentIds={selectedIds} />
      </main>
    </div>
  )
}

function ModeButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors " +
        (active
          ? "bg-primary/10 text-primary"
          : "text-foreground hover:bg-muted")
      }
      aria-current={active ? "page" : undefined}
    >
      {icon}
      {label}
    </button>
  )
}
