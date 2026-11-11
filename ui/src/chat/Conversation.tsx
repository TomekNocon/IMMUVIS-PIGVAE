// The conversation with a connected model: suggested questions, messages and the composer.
import { useEffect, useRef, useState } from 'react'
import type { DrawerProps } from './ChatDrawer'
import { ChatError, type ChatMessage, type Connector } from './connectors'
import { firstCrop } from './crops'
import { Markdown } from './markdown'
import { STARTERS, SYSTEM_PROMPT } from './prompt'

interface Shown extends ChatMessage {
  crop?: { row: number; col: number } // the crop an answer pinned
}

type Props = Omit<DrawerProps, 'result' | 'onClose'> & { connector: Connector; context: object }

export function Conversation({ connector, context, open, pending, onPendingSent, cropRef, pinCrop }: Props) {
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
        {messages.length === 0 && !busy && <Starters onAsk={(q) => void ask(q)} />}
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
          <div role="alert" className="errorText">
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
        <div className="footnote">Enter to send · research use only, not a diagnosis</div>
      </div>
    </>
  )
}

function Starters({ onAsk }: { onAsk: (question: string) => void }) {
  return (
    <div>
      <div className="caps head">Suggested</div>
      <div className="chatStarters">
        {STARTERS.map((q) => (
          <button key={q} onClick={() => onAsk(q)}>
            <span>{q}</span>
            <span className="mut" aria-hidden>
              →
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
