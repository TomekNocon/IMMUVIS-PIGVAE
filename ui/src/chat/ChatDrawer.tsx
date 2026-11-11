// "Ask about this result" (Claude Design handoff): the side drawer with the conversation.
// The model is one the user connects themselves (see connectors.ts); it sees the prediction's
// numbers, never the image.
import { useEffect, useMemo } from 'react'
import type { PredictResponse } from '@/api'
import { cx } from '@/components/cx'
import { Connect } from './Connect'
import { Conversation } from './Conversation'
import type { CropRef } from './markdown'
import { buildContext } from './prompt'
import { setConnected, useConnector } from './store'
import './chat.css'

/** A question from the ask bar; the id makes each click a new question, sent exactly once. */
export interface PendingQuestion {
  text: string
  id: number
}

export interface DrawerProps {
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
    <aside className={cx('chatDrawer', open && 'open')} aria-label="Ask about this result" inert={!open}>
      <div className="chatHead">
        <div className="chatHeadText">
          <div className="caps">Ask about this result</div>
          <div className="mut chatSub">
            {result.prediction} · {(p * 100).toFixed(1)}%
            {connector && (
              <>
                {' · '}
                {connector.label} ·{' '}
                <button className="textbtn" onClick={() => setConnected(null)}>
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
