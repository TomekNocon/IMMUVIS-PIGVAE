import { useMutation } from '@tanstack/react-query'
import { useState, type FormEvent } from 'react'
import { api, ApiError, USING_MOCK } from '@/api'

function loginMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 401) return 'Wrong password.'
    if (err.status === 429) return 'Too many attempts, try again in a few minutes.'
    return err.message
  }
  return 'Something went wrong. Try again.'
}

export function Login({ note, onDone }: { note: string | null; onDone: () => void }) {
  const [password, setPassword] = useState('')
  const login = useMutation({ mutationFn: api.login, onSuccess: onDone })

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (password) login.mutate(password)
  }

  return (
    <main style={{ flex: 1, display: 'grid', placeItems: 'center', padding: '32px 20px' }}>
      <form
        onSubmit={submit}
        className="rise"
        style={{ width: '100%', maxWidth: 380, display: 'flex', flexDirection: 'column', gap: 16 }}
      >
        <h1 className="serif" style={{ fontWeight: 400, fontSize: 40, lineHeight: 1.1, margin: 0 }}>
          Sign in
        </h1>
        {note && (
          <div
            className="mut"
            style={{ fontSize: 14, border: '1px solid var(--line)', borderRadius: 6, padding: '10px 12px' }}
          >
            {note}
          </div>
        )}
        <label className="mut" style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 13 }}>
          Password
          <input
            className="input"
            type="password"
            autoFocus
            autoComplete="current-password"
            value={password}
            onChange={(e) => {
              setPassword(e.target.value)
              login.reset()
            }}
          />
        </label>
        {login.error && (
          <div role="alert" style={{ fontSize: 14, color: 'var(--err)' }}>
            {loginMessage(login.error)}
          </div>
        )}
        <button className="primary" type="submit" disabled={login.isPending}>
          {login.isPending ? 'Signing in…' : 'Sign in'}
        </button>
        {USING_MOCK && (
          <div className="mono mut" style={{ fontSize: 12 }}>
            Mock: any password works. "wrong" gives 401, "locked" gives 429.
          </div>
        )}
      </form>
    </main>
  )
}
