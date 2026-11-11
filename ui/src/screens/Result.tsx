import { useCallback, useEffect, useMemo, useState } from 'react'
import type { InfoResponse, PredictResponse } from '@/api'
import { AskBar } from '@/chat/AskBar'
import { ChatDrawer, type PendingQuestion } from '@/chat/ChatDrawer'
import type { CropRef } from '@/chat/markdown'
import { buildModel } from '@/result/model'
import { Panel } from '@/result/Panel'
import { Viewer } from '@/result/Viewer'
import './Result.css'

export function Result({
  result,
  info,
  onAgain,
}: {
  result: PredictResponse
  info: InfoResponse | undefined
  onAgain: () => void
}) {
  const m = useMemo(() => buildModel(result), [result])
  const [hover, setHover] = useState<number | null>(null)
  const [pin, setPin] = useState<number | null>(null)
  const [chatOpen, setChatOpen] = useState(false)
  const [pending, setPending] = useState<PendingQuestion | null>(null)
  const togglePin = useCallback((i: number) => setPin((p) => (p === i ? null : i)), [])
  const closeChat = useCallback(() => setChatOpen(false), [])
  const pendingSent = useCallback(() => setPending(null), [])

  // On wide screens the whole page makes room for the open drawer (see .app in app.css).
  useEffect(() => {
    document.body.classList.toggle('chatOpen', chatOpen)
    return () => document.body.classList.remove('chatOpen')
  }, [chatOpen])

  const cropIndex = useMemo(() => new Map(m.crops.map((c) => [`${c.row},${c.col}`, c.i])), [m])
  const pinCrop = useCallback(
    (row: number, col: number) => {
      const i = cropIndex.get(`${row},${col}`)
      if (i == null) return false
      setPin(i)
      return true
    },
    [cropIndex],
  )

  // "row R, col C" in an answer links to that crop: hover highlights it, click pins it.
  const cropRef = useMemo<CropRef>(
    () => (row, col, label, key) => {
      const i = cropIndex.get(`${row},${col}`)
      if (i == null) return null
      return (
        <button
          key={key}
          className="cropRef"
          onMouseEnter={() => setHover(i)}
          onMouseLeave={() => setHover(null)}
          onFocus={() => setHover(i)}
          onBlur={() => setHover(null)}
          onClick={() => togglePin(i)}
        >
          {label}
        </button>
      )
    },
    [cropIndex, togglePin],
  )

  return (
    <>
      <main className="resultMain">
        <Viewer m={m} hover={hover} pin={pin} setHover={setHover} togglePin={togglePin} />
        <div className="resultSide">
          <Panel
            m={m}
            hover={hover}
            pin={pin}
            setHover={setHover}
            togglePin={togglePin}
            onAgain={onAgain}
            ask={
              <AskBar
                onAsk={(q) => {
                  setPending({ text: q, id: Date.now() })
                  setChatOpen(true)
                }}
                onOpen={() => setChatOpen(true)}
              />
            }
          />
          <div className="footnote">
            {result.model.label} · fold {result.model.fold} · AUROC {result.model.fold_val_auroc.toFixed(3)} ·{' '}
            {result.model.bundle}
            {info ? ` · ${info.device}` : ''}
          </div>
        </div>
      </main>
      <ChatDrawer
        result={result}
        open={chatOpen}
        onClose={closeChat}
        pending={pending}
        onPendingSent={pendingSent}
        cropRef={cropRef}
        pinCrop={pinCrop}
      />
    </>
  )
}
