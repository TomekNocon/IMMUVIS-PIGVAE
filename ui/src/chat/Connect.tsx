// Choosing and connecting a language model: the local bridge, an OpenAI key, or demo replies.
import { useState, type FormEvent } from 'react'
import { USING_MOCK } from '@/api'
import { cx } from '@/components/cx'
import { Field } from '@/components/Field'
import type { PendingQuestion } from './ChatDrawer'
import {
  ChatError,
  connectBridge,
  connectOpenAI,
  demoConnector,
  forgetKey,
  loadKey,
  loadSettings,
  saveKey,
  saveSettings,
  type ChatSettings,
  type Connector,
} from './connectors'
import { setConnected } from './store'

type Kind = ChatSettings['kind']

const KINDS: [Kind, string][] = [
  ['bridge', 'My subscription'],
  ['openai', 'OpenAI API key'],
  ...(USING_MOCK ? [['demo', 'Demo'] as [Kind, string]] : []),
]

export function Connect({ pending }: { pending: PendingQuestion | null }) {
  const [s, setS] = useState<ChatSettings>(() => {
    const loaded = loadSettings()
    return !USING_MOCK && loaded.kind === 'demo' ? { ...loaded, kind: 'bridge' } : loaded
  })
  const [key, setKey] = useState(loadKey)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const update = (patch: Partial<ChatSettings>) => setS((old) => ({ ...old, ...patch }))

  const connect = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      let c: Connector
      if (s.kind === 'bridge') c = await connectBridge(s.bridgeUrl)
      else if (s.kind === 'openai') {
        c = await connectOpenAI(key.trim(), s.model.trim())
        saveKey(key.trim(), s.remember)
      } else c = demoConnector()
      saveSettings(s)
      setConnected(c)
    } catch (err) {
      setError(err instanceof ChatError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={connect} className="chatConnect">
      {pending && <div className="chatMsg user">{pending.text}</div>}
      <p>
        Connect a language model to answer{pending ? ' this' : ''}. It reads only the numbers on this page, never
        the image.
      </p>
      <div className="chatSeg" role="group" aria-label="Connection">
        {KINDS.map(([k, label]) => (
          <button
            key={k}
            type="button"
            className={cx('chatSegBtn', s.kind === k && 'on')}
            aria-pressed={s.kind === k}
            onClick={() => update({ kind: k })}
          >
            {label}
          </button>
        ))}
      </div>

      {s.kind === 'bridge' && (
        <>
          <div className="mut chatNote">
            Uses your ChatGPT (Codex CLI) or Claude (Claude Code) plan through a small bridge on your computer,
            signed in to one of the CLIs:
            <pre className="mono chatCode">make chat-bridge</pre>
          </div>
          <Field label="Bridge address">
            <input className="input chatField" value={s.bridgeUrl} onChange={(e) => update({ bridgeUrl: e.target.value })} />
          </Field>
        </>
      )}

      {s.kind === 'openai' && (
        <>
          <Field label="API key (platform.openai.com; a ChatGPT plan is not an API key)">
            <input
              className="input chatField"
              type="password"
              autoComplete="off"
              placeholder="sk-..."
              value={key}
              onChange={(e) => setKey(e.target.value)}
            />
          </Field>
          <Field label="Model">
            <input className="input chatField mono" value={s.model} onChange={(e) => update({ model: e.target.value })} />
          </Field>
          <label className="chatCheck">
            <input type="checkbox" checked={s.remember} onChange={(e) => update({ remember: e.target.checked })} />
            Remember the key on this device
          </label>
          <div className="mut chatKeyNote">
            The key stays in this browser and goes only to OpenAI, which bills it.{' '}
            {key && (
              <button
                type="button"
                className="textbtn"
                onClick={() => {
                  forgetKey()
                  setKey('')
                }}
              >
                forget key
              </button>
            )}
          </div>
        </>
      )}

      {s.kind === 'demo' && <div className="mut chatNote">Canned answers, to try the chat without a model.</div>}

      {error && (
        <div role="alert" className="errorText">
          {error}
        </div>
      )}
      <button className="primary compact" disabled={busy || (s.kind === 'openai' && (!key.trim() || !s.model.trim()))}>
        {busy ? 'Connecting…' : 'Connect'}
      </button>
    </form>
  )
}
