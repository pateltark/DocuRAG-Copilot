"use client"

import { useEffect, useState } from "react"
import { ChatSession, listChatSessions } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Plus, MessageSquare } from "lucide-react"

type ChatSidebarProps = {
    mode: "sec" | "doc"
    activeChatId: string | null
    onSelectChat: (chatId: string) => void
    onNewChat: () => void
    refreshKey?: number // bump this after sending a message to re-pull the list
}

export function ChatSidebar({
    mode,
    activeChatId,
    onSelectChat,
    onNewChat,
    refreshKey,
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
        <div className="flex flex-col h-full w-64 border-r bg-muted/20">
            <div className="p-2 border-b">
                <Button onClick={onNewChat} variant="outline" className="w-full justify-start gap-2">
                    <Plus size={16} />
                    New chat
                </Button>
            </div>

            <div className="flex-1 overflow-y-auto p-1">
                {loading && (
                    <p className="text-xs text-muted-foreground px-2 py-2">Loading…</p>
                )}

                {!loading && sessions.length === 0 && (
                    <p className="text-xs text-muted-foreground px-2 py-4">
                        No chats yet. Start one below.
                    </p>
                )}

                {sessions.map((s) => (
                    <button
                        key={s.chat_id}
                        onClick={() => onSelectChat(s.chat_id)}
                        className={`w-full text-left flex items-start gap-2 px-2 py-2 rounded-md text-sm truncate transition-colors ${activeChatId === s.chat_id
                                ? "bg-muted font-medium"
                                : "hover:bg-muted/60 text-muted-foreground"
                            }`}
                    >
                        <MessageSquare size={14} className="mt-0.5 shrink-0" />
                        <span className="truncate">{s.title}</span>
                    </button>
                ))}
            </div>
        </div>
    )
}