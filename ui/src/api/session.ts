// The session token, kept in memory and in sessionStorage (gone when the tab closes).
const KEY = 'pigvae-session'

interface Session {
  token: string
  expiresAt: number // unix seconds
}

let current: Session | null = load()

function load(): Session | null {
  try {
    const raw = sessionStorage.getItem(KEY)
    return raw ? (JSON.parse(raw) as Session) : null
  } catch {
    return null
  }
}

export function setSession(token: string, expiresAt: number): void {
  current = { token, expiresAt }
  try {
    sessionStorage.setItem(KEY, JSON.stringify(current))
  } catch {
    // storage blocked: the in-memory session still works for this page
  }
}

export function clearSession(): void {
  current = null
  try {
    sessionStorage.removeItem(KEY)
  } catch {
    // ignore
  }
}

/** The token if the session has not expired, else null. */
export function getToken(): string | null {
  if (!current || current.expiresAt * 1000 <= Date.now()) return null
  return current.token
}
