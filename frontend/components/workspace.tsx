"use client"

import { useEffect, useState } from "react"
import { ChatPanel } from "@/components/chat-panel"
import { ChatSidebar } from "@/components/chat-sidebar"
import { DocumentManager } from "@/components/document-manager"
import { UserMenu } from "@/components/user-menu"
import { FileSearch, FileText, Landmark } from "lucide-react"

type Mode = "sec" | "doc"

export function Workspace() {
  const [mode, setMode] = useState<Mode>("sec")
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [activeChatId, setActiveChatId] = useState<string | null>(null)
  const [sidebarRefreshKey, setSidebarRefreshKey] = useState(0)

  // Switching modes means switching chat lists — start fresh each time.
  useEffect(() => {
    setActiveChatId(null)
  }, [mode])

  function handleChatIdChange(id: string | null) {
    setActiveChatId(id)
    // A message was sent (id is non-null) — the sidebar's list/title/order
    // may now be stale, so refetch it. No need to bump on `null` (a clear),
    // since a cleared chat_id disappears from the list on its own next load.
    if (id) setSidebarRefreshKey((k) => k + 1)
  }

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
          <UserMenu />
        </div>
      </aside>

      {/* Chat sessions list for the current mode */}
      <ChatSidebar
        mode={mode}
        activeChatId={activeChatId}
        onSelectChat={setActiveChatId}
        onNewChat={() => setActiveChatId(null)}
        refreshKey={sidebarRefreshKey}
      />

      <main className="flex-1 overflow-hidden">
        {/* Keying by mode remounts the panel so history reloads per mode.
            chatId itself is handled inside ChatPanel via the SWR key, so it
            doesn't need to be part of this key. */}
        <ChatPanel
          key={mode}
          mode={mode}
          selectedDocumentIds={selectedIds}
          chatId={activeChatId}
          onChatIdChange={handleChatIdChange}
        />
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