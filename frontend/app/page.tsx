"use client"

import { AuthForm } from "@/components/auth-form"
import { Workspace } from "@/components/workspace"
import { useAuth } from "@/lib/auth-context"
import { Loader2 } from "lucide-react"

export default function Page() {
  const { ready, isAuthenticated } = useAuth()

  if (!ready) {
    return (
      <main className="flex min-h-svh items-center justify-center bg-background">
        <Loader2 className="size-5 animate-spin text-muted-foreground" aria-hidden="true" />
        <span className="sr-only">Loading</span>
      </main>
    )
  }

  return isAuthenticated ? <Workspace /> : <AuthForm />
}
