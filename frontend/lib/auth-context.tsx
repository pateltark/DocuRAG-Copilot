"use client"

import { createContext, useContext, useEffect, useState, type ReactNode } from "react"
import { clearToken, getToken, login as apiLogin, register as apiRegister, setToken } from "@/lib/api"

type AuthContextValue = {
  token: string | null
  ready: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  register: (input: { email: string; password: string; name: string }) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setTokenState] = useState<string | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    setTokenState(getToken())
    setReady(true)
  }, [])

  async function login(email: string, password: string) {
    const res = await apiLogin({ email, password })
    setToken(res.access_token)
    setTokenState(res.access_token)
  }

  async function register(input: { email: string; password: string; name: string }) {
    const user_id =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(36).slice(2)}`
    await apiRegister({ ...input, user_id })
    // Auto-login after successful registration.
    await login(input.email, input.password)
  }

  function logout() {
    clearToken()
    setTokenState(null)
  }

  return (
    <AuthContext.Provider
      value={{
        token,
        ready,
        isAuthenticated: Boolean(token),
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider")
  return ctx
}
