// "Ask about this result" (Claude Design handoff): quick questions under the prediction (AskBar)
// and a side drawer with the conversation (ChatDrawer). The model is one the user connects
// themselves (see connectors.ts); it sees the prediction's numbers, never the image.
import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type FormEvent, type ReactNode } from 'react'
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
import { firstCrop } from './crops'
import { Markdown, type CropRef } from './markdown'
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
    (l) => {
      listeners.add(l)
      return () => {
        listeners.delete(l)
      }
    },
    () => connected,
  )

const QUICK: [string, number][] = [
  ['Why this prediction?', 0],
  ['Most important crop', 1],
  ['How confident?', 2],
]

/** Quick questions under the prediction; each opens the drawer and asks. */
export function AskBar({ onAsk, onOpen }: { onAsk: (question: string) => void; onOpen: () => void }) {
  return (
    <div>
      <div className="caps" style={{ marginBottom: 8 }}>
        Ask about this result
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 18px', fontSize: 14 }}>
        {QUICK.map(([label, i]) => (
          <button key={label} className="askLink" onClick={() => onAsk(STARTERS[i])}>
            {label}
          </button>
        ))}
        <button className="askLink own" onClick={onOpen}>
          Ask your own →
        </button>
      </div>
    </div>
  )
}

/** A question from the ask bar; the id makes each click a new question, sent exactly once. */
export interface PendingQuestion {
  text: string
  id: number
}

interface DrawerProps {
  result: PredictResponse
  open: boolean
  onClose: () => void
  /** A question from the ask bar, sent as soon as a model is connected. */
  pending: PendingQuestion | null
  onPendingSent: () => void
  cropRef: CropRef
  /** Pin a crop by position (answers that mention a crop pin it). */
  pinCrop: (row: number, col: number) => boolean
}

export function ChatDrawer(props: DrawerProps) {
  const { result, open, onClose } = props
  const connector = useConnector()
  const context = useMemo(() => buildContext(result), [result])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  const p = result.probabilities[result.prediction] ?? 0
  return (
    <aside className={`chatDrawer${open ? ' open' : ''}`} aria-label="Ask about this result" inert={!open}>
      <div className="chatHead">
        <div style={{ minWidth: 0 }}>
          <div className="caps">Ask about this result</div>
          <div className="mut" style={{ fontSize: 13, marginTop: 2, overflowWrap: 'anywhere' }}>
            {result.prediction} · {(p * 100).toFixed(1)}%
            {connector && (
              <>
                {' · '}
                {connector.label} ·{' '}
                <button className="textbtn" style={{ fontSize: 13 }} onClick={() => setConnected(null)}>
                  disconnect
                </button>
              </>
            )}
          </div>
        </div>
        <button className="chatClose" aria-label="Close chat" onClick={onClose}>
          ×
        </button>
      </div>
      {connector ? (
        <Conversation {...props} connector={connector} context={context} />
      ) : (
        <div className="chatBody">
          <Connect pending={props.pending} />
        </div>
      )}
    </aside>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="mut" style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 13 }}>
      {label}
      {children}
    </label>
  )
}

function Connect({ pending }: { pending: PendingQuestion | null }) {
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
    ['bridge', 'My subscription'],
    ['openai', 'OpenAI API key'],
    ...(USING_MOCK ? [['demo', 'Demo'] as [ChatSettings['kind'], string]] : []),
  ]

  return (
    <form onSubmit={connect} style={{ display: 'flex', flexDirection: 'column', gap: 16, fontSize: 14 }}>
      {pending && (
        <div className="chatMsg user" style={{ alignSelf: 'flex-end' }}>
          {pending.text}
        </div>
      )}
      <p style={{ margin: 0 }}>
        Connect a language model to answer{pending ? ' this' : ''}. It reads only the numbers on this page, never
        the image.
      </p>
      <div className="chatSeg" role="group" aria-label="Connection">
        {kinds.map(([k, label]) => (
          <button
            key={k}
            type="button"
            className={`chatSegBtn${s.kind === k ? ' on' : ''}`}
            aria-pressed={s.kind === k}
            onClick={() => update({ kind: k })}
          >
            {label}
          </button>
        ))}
      </div>

      {s.kind === 'bridge' && (
        <>
          <div className="mut" style={{ fontSize: 13 }}>
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
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
            <input type="checkbox" checked={s.remember} onChange={(e) => update({ remember: e.target.checked })} />
            Remember the key on this device
          </label>
          <div className="mut" style={{ fontSize: 12 }}>
            The key stays in this browser and goes only to OpenAI, which bills it.{' '}
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

      {s.kind === 'demo' && <div className="mut" style={{ fontSize: 13 }}>Canned answers, to try the chat without a model.</div>}

      {error && (
        <div role="alert" style={{ color: 'var(--err)', fontSize: 14 }}>
          {error}
        </div>
      )}
      <button
        className="primary"
        style={{ height: 44, alignSelf: 'flex-start', padding: '0 22px' }}
        disabled={busy || (s.kind === 'openai' && (!key.trim() || !s.model.trim()))}
      >
        {busy ? 'Connecting…' : 'Connect'}
      </button>
    </form>
  )
}

interface Shown extends ChatMessage {
  crop?: { row: number; col: number } // the crop an answer pinned
}

function Conversation({
  connector,
  context,
  open,
  pending,
  onPendingSent,
  cropRef,
  pinCrop,
}: DrawerProps & { connector: Connector; context: object }) {
  const [messages, setMessages] = useState<Shown[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const sentId = useRef<number | null>(null)

  useEffect(() => {
    const el = bodyRef.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [messages, busy])
  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  const ask = async (text: string) => {
    const question = text.trim()
    if (!question || busy) return
    const history: ChatMessage[] = [...messages.map(({ role, content }) => ({ role, content })), { role: 'user', content: question }]
    setMessages((m) => [...m, { role: 'user', content: question }])
    setDraft('')
    setBusy(true)
    setError(null)
    try {
      const reply = await connector.send(SYSTEM_PROMPT, context, history)
      // An answer about a crop pins it in the viewer, as in the design.
      const ref = firstCrop(reply)
      const crop = ref && pinCrop(ref.row, ref.col) ? ref : undefined
      setMessages((m) => [...m, { role: 'assistant', content: reply, crop }])
    } catch (err) {
      setError(err instanceof ChatError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  // A question from the ask bar: send it once (React runs effects twice in development).
  useEffect(() => {
    if (!pending || sentId.current === pending.id) return
    sentId.current = pending.id
    onPendingSent()
    void ask(pending.text)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending])

  return (
    <>
      <div className="chatBody" ref={bodyRef} aria-live="polite">
        {messages.length === 0 && !busy && (
          <div>
            <div className="caps" style={{ marginBottom: 8 }}>
              Suggested
            </div>
            <div className="chatStarters">
              {STARTERS.map((q) => (
                <button key={q} onClick={() => void ask(q)}>
                  <span>{q}</span>
                  <span className="mut" aria-hidden>
                    →
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className="chatTurn">
            {m.role === 'user' ? (
              <div className="chatMsg user">{m.content}</div>
            ) : (
              <>
                <div className="chatMsg reply">
                  <Markdown text={m.content} cropRef={cropRef} />
                </div>
                {m.crop && (
                  <button className="chatPin" onClick={() => pinCrop(m.crop!.row, m.crop!.col)}>
                    pinned · row {m.crop.row}, col {m.crop.col}
                  </button>
                )}
              </>
            )}
          </div>
        ))}
        {busy && (
          <div className="chatThinking" aria-label="Thinking">
            <span />
            <span />
            <span />
          </div>
        )}
        {error && (
          <div role="alert" style={{ color: 'var(--err)', fontSize: 14 }}>
            {error}
          </div>
        )}
      </div>
      <div className="chatFoot">
        <form
          className="chatComposer"
          onSubmit={(e) => {
            e.preventDefault()
            void ask(draft)
          }}
        >
          <input
            ref={inputRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Ask about this prediction…"
            aria-label="Your question"
          />
          <button type="submit" className="chatSend" aria-label="Send" disabled={busy || !draft.trim()}>
            →
          </button>
        </form>
        <div className="mono mut" style={{ fontSize: 11 }}>
          Enter to send · research use only, not a diagnosis
        </div>
      </div>
    </>
  )
}
