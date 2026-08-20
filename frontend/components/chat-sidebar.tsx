"use client"

import { useEffect, useState } from "react"
import { ChatSession, listChatSessions } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Plus, MessageSquare, PanelLeftClose, PanelLeftOpen } from "lucide-react"

type ChatSidebarProps = {
    mode: "sec" | "doc"
    activeChatId: string | null
    onSelectChat: (chatId: string) => void
    onNewChat: () => void
    refreshKey?: number // bump this after sending a message to re-pull the list
    collapsed: boolean
    onToggleCollapsed: () => void
}

export function ChatSidebar({
    mode,
    activeChatId,
    onSelectChat,
    onNewChat,
    refreshKey,
    collapsed,
    onToggleCollapsed,
}: ChatSidebarProps) {
    const [sessions, setSessions] = useState<ChatSession[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        let cancelled = false
        setLoading(true)
        listChatSessions(mode)
            .then((data) => {
                if (!cancelled) setSessions(data)
            })
            .finally(() => {
                if (!cancelled) setLoading(false)
            })
        return () => {
            cancelled = true
        }
    }, [mode, refreshKey])

    return (
        // Outer wrapper does NOT clip — the toggle button lives here so it's
        // never a child of the part that collapses to width:0.
        <div className="relative h-full shrink-0">
            {/* Toggle handle — always rendered, always visible, in both states */}
            <button
                onClick={onToggleCollapsed}
                aria-label={collapsed ? "Show chat history" : "Hide chat history"}
                title={collapsed ? "Show chat history" : "Hide chat history"}
                className="absolute top-4 z-20 flex size-7 items-center justify-center rounded-full border border-border bg-background shadow-sm hover:bg-muted transition-[left] duration-200 ease-in-out"
                style={{ left: collapsed ? "0.5rem" : "16rem" /* 16rem = w-64, sits on the edge */, transform: "translateX(-50%)" }}
            >
                {collapsed ? <PanelLeftOpen size={14} /> : <PanelLeftClose size={14} />}
            </button>

            {/* The actual panel — THIS is what collapses */}
            <div
                className={`h-full overflow-hidden border-r bg-muted/20 transition-[width] duration-200 ease-in-out ${collapsed ? "w-0 border-r-0" : "w-64"
                    }`}
            >
                <div className="flex h-full w-64 flex-col">
                    <div className="border-b p-2">
                        <Button onClick={onNewChat} variant="outline" className="w-full justify-start gap-2">
                            <Plus size={16} />
                            New chat
                        </Button>
                    </div>

                    <div className="flex-1 overflow-y-auto p-1">
                        {loading && <p className="px-2 py-2 text-xs text-muted-foreground">Loading…</p>}

                        {!loading && sessions.length === 0 && (
                            <p className="px-2 py-4 text-xs text-muted-foreground">No chats yet. Start one below.</p>
                        )}

                        {sessions.map((s) => (
                            <button
                                key={s.chat_id}
                                onClick={() => onSelectChat(s.chat_id)}
                                className={`flex w-full items-start gap-2 truncate rounded-md px-2 py-2 text-left text-sm transition-colors ${activeChatId === s.chat_id
                                        ? "bg-muted font-medium"
                                        : "text-muted-foreground hover:bg-muted/60"
                                    }`}
                            >
                                <MessageSquare size={14} className="mt-0.5 shrink-0" />
                                <span className="truncate">{s.title}</span>
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}