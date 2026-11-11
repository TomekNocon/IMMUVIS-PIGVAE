import { useMutation } from '@tanstack/react-query'
import { useState, type FormEvent } from 'react'
import { api, ApiError, USING_MOCK } from '@/api'
import { Field } from '@/components/Field'
import './Login.css'

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
    <main className="login">
      <form onSubmit={submit} className="rise loginForm">
        <h1 className="title">Sign in</h1>
        {note && <div className="mut loginNote">{note}</div>}
        <Field label="Password">
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
        </Field>
        {login.error && (
          <div role="alert" className="errorText">
            {loginMessage(login.error)}
          </div>
        )}
        <button className="primary" type="submit" disabled={login.isPending}>
          {login.isPending ? 'Signing in…' : 'Sign in'}
        </button>
        {USING_MOCK && <div className="hint">Mock: any password works. "wrong" gives 401, "locked" gives 429.</div>}
      </form>
    </main>
  )
}
