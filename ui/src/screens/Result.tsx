import { useMemo, useState } from 'react'
import type { InfoResponse, PredictResponse } from '@/api'
import { buildModel } from '@/result/model'
import { Panel } from '@/result/Panel'
import { Viewer } from '@/result/Viewer'

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
  const togglePin = (i: number) => setPin((p) => (p === i ? null : i))

  return (
    <main
      style={{
        flex: 1, width: '100%', maxWidth: 1320, margin: '0 auto', padding: '24px 28px 40px', display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 340px), 1fr))', gap: 28, alignItems: 'start',
      }}
    >
      <Viewer m={m} hover={hover} pin={pin} setHover={setHover} togglePin={togglePin} />
      <div style={{ display: 'flex', flexDirection: 'column', gap: 28, minWidth: 0 }}>
        <Panel m={m} hover={hover} pin={pin} setHover={setHover} togglePin={togglePin} onAgain={onAgain} />
        <div className="mono mut" style={{ fontSize: 11 }}>
          {result.model.label} · fold {result.model.fold} · AUROC {result.model.fold_val_auroc.toFixed(3)} ·{' '}
          {result.model.bundle}
          {info ? ` · ${info.device}` : ''}
        </div>
      </div>
    </main>
  )
}
