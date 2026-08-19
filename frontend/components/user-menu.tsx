"use client"

import { useEffect, useRef, useState } from "react"
import { useAuth } from "@/lib/auth-context"
import { LogOut } from "lucide-react"

function getInitials(name: string | undefined): string {
    if (!name) return "?"
    const parts = name.trim().split(/\s+/)
    const first = parts[0]?.[0] ?? ""
    const last = parts.length > 1 ? parts[parts.length - 1][0] : ""
    return (first + last).toUpperCase() || "?"
}

export function UserMenu() {
    const { user, logout } = useAuth()
    const [open, setOpen] = useState(false)
    const ref = useRef<HTMLDivElement>(null)

    useEffect(() => {
        function onClickOutside(e: MouseEvent) {
            if (ref.current && !ref.current.contains(e.target as Node)) {
                setOpen(false)
            }
        }
        document.addEventListener("mousedown", onClickOutside)
        return () => document.removeEventListener("mousedown", onClickOutside)
    }, [])

    return (
        <div ref={ref} className="relative">
            <button
                type="button"
                onClick={() => setOpen((v) => !v)}
                className="flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left hover:bg-muted"
                aria-haspopup="true"
                aria-expanded={open}
            >
                <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary text-xs font-semibold text-primary-foreground">
                    {getInitials(user?.name)}
                </div>
                <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">{user?.name ?? "Loading…"}</p>
                    <p className="truncate text-xs text-muted-foreground">{user?.email ?? ""}</p>
                </div>
            </button>

            {open && (
                <div className="absolute bottom-full left-0 mb-2 w-full min-w-56 rounded-lg border border-border bg-card p-1 shadow-md">
                    <div className="border-b border-border px-3 py-2">
                        <p className="truncate text-sm font-medium">{user?.name ?? "—"}</p>
                        <p className="truncate text-xs text-muted-foreground">{user?.email ?? "—"}</p>
                    </div>
                    <button
                        type="button"
                        onClick={logout}
                        className="mt-1 flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-foreground hover:bg-muted"
                    >
                        <LogOut className="size-4" aria-hidden="true" />
                        Sign out
                    </button>
                </div>
            )}
        </div>
    )
}