// "Ask about this result": optional chat with a language model the user connects themselves.
import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type FormEvent, type KeyboardEvent } from 'react'
import { USING_MOCK, type PredictResponse } from '@/api'
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
  type ChatMessage,
  type ChatSettings,
  type Connector,
} from './connectors'
import { Markdown } from './markdown'
import { buildContext, STARTERS, SYSTEM_PROMPT } from './prompt'

// The connection outlives one result ("Analyse another image" keeps it).
let connected: Connector | null = null
const listeners = new Set<() => void>()
const setConnected = (c: Connector | null) => {
  connected = c
  listeners.forEach((l) => l())
}
const useConnector = () =>
  useSyncExternalStore(
    (l) => (listeners.add(l), () => listeners.delete(l)),
    () => connected,
  )

export function Chat({ result }: { result: PredictResponse }) {
  const connector = useConnector()
  const context = useMemo(() => buildContext(result), [result])
  return (
    <section className="chat">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
        <span className="caps">Ask about this result</span>
        {connector && (
          <span className="mut" style={{ fontSize: 12 }}>
            {connector.label} ·{' '}
            <button className="textbtn" style={{ fontSize: 12 }} onClick={() => setConnected(null)}>
              disconnect
            </button>
          </span>
        )}
      </div>
      {connector ? <Conversation connector={connector} context={context} /> : <Connect />}
    </section>
  )
}

function Connect() {
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

  const kinds: [ChatSettings['kind'], string][] = [
    ['bridge', 'My subscription (local)'],
    ['openai', 'OpenAI API key'],
    ...(USING_MOCK ? [['demo', 'Demo replies'] as [ChatSettings['kind'], string]] : []),
  ]

  return (
    <form onSubmit={connect} className="card" style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12, fontSize: 14 }}>
      <p style={{ margin: 0 }} className="mut">
        Chat about this prediction with a language model you connect. It reads only the numbers on this page,
        never the image.
      </p>
      <div className="chatTabs" role="group" aria-label="Connection">
        {kinds.map(([k, label]) => (
          <button key={k} type="button" className={`chatTab${s.kind === k ? ' on' : ''}`} aria-pressed={s.kind === k} onClick={() => update({ kind: k })}>
            {label}
          </button>
        ))}
      </div>

      {s.kind === 'bridge' && (
        <>
          <div className="mut" style={{ fontSize: 13 }}>
            Uses your ChatGPT (Codex CLI) or Claude (Claude Code) subscription through a small bridge on your computer.
            Start it there, signed in to one of the CLIs:
            <pre className="mono chatCode">python3 scripts/deploy/chat_bridge.py</pre>
          </div>
          <label className="mut" style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 13 }}>
            Bridge address
            <input className="input" style={{ height: 40, fontSize: 14 }} value={s.bridgeUrl} onChange={(e) => update({ bridgeUrl: e.target.value })} />
          </label>
        </>
      )}

      {s.kind === 'openai' && (
        <>
          <label className="mut" style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 13 }}>
            API key (from platform.openai.com; a ChatGPT subscription is not an API key)
            <input
              className="input"
              style={{ height: 40, fontSize: 14 }}
              type="password"
              autoComplete="off"
              placeholder="sk-..."
              value={key}
              onChange={(e) => setKey(e.target.value)}
            />
          </label>
          <label className="mut" style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 13 }}>
            Model
            <input className="input mono" style={{ height: 40, fontSize: 14 }} value={s.model} onChange={(e) => update({ model: e.target.value })} />
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
            <input type="checkbox" checked={s.remember} onChange={(e) => update({ remember: e.target.checked })} />
            Remember the key on this device (otherwise it is forgotten when the tab closes)
          </label>
          <div className="mut" style={{ fontSize: 12 }}>
            The key stays in this browser and is sent only to OpenAI, which bills it.{' '}
            {key && (
              <button
                type="button"
                className="textbtn"
                style={{ fontSize: 12 }}
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

      {s.kind === 'demo' && <div className="mut" style={{ fontSize: 13 }}>Canned answers, to try the panel without a model.</div>}

      {error && (
        <div role="alert" style={{ color: 'var(--err)', fontSize: 14 }}>
          {error}
        </div>
      )}
      <button
        className="primary"
        style={{ height: 40, alignSelf: 'flex-start', padding: '0 18px' }}
        disabled={busy || (s.kind === 'openai' && (!key.trim() || !s.model.trim()))}
      >
        {busy ? 'Connecting…' : 'Connect'}
      </button>
    </form>
  )
}

function Conversation({ connector, context }: { connector: Connector; context: object }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Braces: newer browsers return a Promise from scrollIntoView, which must not become the cleanup.
    endRef.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [messages, busy])

  const ask = async (text: string) => {
    const question = text.trim()
    if (!question || busy) return
    const next: ChatMessage[] = [...messages, { role: 'user', content: question }]
    setMessages(next)
    setDraft('')
    setBusy(true)
    setError(null)
    try {
      const reply = await connector.send(SYSTEM_PROMPT, context, next)
      setMessages([...next, { role: 'assistant', content: reply }])
    } catch (err) {
      setError(err instanceof ChatError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void ask(draft)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div className="chatLog" aria-live="polite">
        {messages.length === 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {STARTERS.map((q) => (
              <button key={q} className="chatChip" onClick={() => void ask(q)}>
                {q}
              </button>
            ))}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={m.role === 'user' ? 'chatMsg user' : 'chatMsg reply'}>
            {m.role === 'user' ? m.content : <Markdown text={m.content} />}
          </div>
        ))}
        {busy && (
          <div className="mut" style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--a)', animation: 'pulse 1.2s infinite' }} />
            Thinking…
          </div>
        )}
        {error && (
          <div role="alert" style={{ color: 'var(--err)', fontSize: 14 }}>
            {error}
          </div>
        )}
        <div ref={endRef} />
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          void ask(draft)
        }}
        style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}
      >
        <textarea
          className="input chatInput"
          rows={2}
          placeholder="Ask about this prediction… (Enter to send, Shift+Enter for a new line)"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKey}
        />
        <button className="primary" style={{ height: 44, padding: '0 18px' }} disabled={busy || !draft.trim()}>
          Send
        </button>
      </form>
      <div className="mono mut" style={{ fontSize: 11 }}>
        Answers are generated by a language model from the numbers on this page. Research use only, not a diagnosis.
      </div>
    </div>
  )
}
