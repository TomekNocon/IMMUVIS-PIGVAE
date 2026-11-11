import { useEffect, useState } from 'react'
import { cx } from '@/components/cx'
import './Predicting.css'

const STAGES = [
  'Preprocessing channels',
  'Embedding cells',
  'Encoding tissue graph',
  'Weighing crops',
  'Attributing the decision',
]
const CUTS = [0.08, 0.55, 0.72, 0.8, 1] // stage ends, as a fraction of the expected time
const TILES = 20

/** Time-based estimate only: the server does not report progress. Holds at 97% if it runs long. */
export function Predicting({ fileName, expectedSeconds }: { fileName: string; expectedSeconds: number }) {
  const [t0] = useState(() => Date.now())
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const id = setInterval(() => {
      setProgress(Math.min(0.97, (Date.now() - t0) / (expectedSeconds * 1000)))
    }, 100)
    return () => clearInterval(id)
  }, [t0, expectedSeconds])

  const percent = Math.round(progress * 100)
  return (
    <main className="predicting">
      <div className="rise predictingBody">
        <TileGrid lit={Math.floor(progress * (TILES + 1))} />
        <div className="predictingSide">
          <div>
            <h1 className="title">Reading the tissue</h1>
            <div className="mut predictingFile">
              {fileName} · usually about {expectedSeconds} seconds
            </div>
          </div>
          <div role="progressbar" aria-valuenow={percent} className="progress">
            <div style={{ width: `${percent}%` }} />
          </div>
          <StageList active={CUTS.findIndex((c) => progress < c)} />
          <div className="hint">Stages are an estimate. The server does not report progress.</div>
        </div>
      </div>
    </main>
  )
}

/** A grid of crops lighting up one by one, the current one brightest. */
function TileGrid({ lit }: { lit: number }) {
  return (
    <div aria-hidden className="tileGrid">
      {Array.from({ length: TILES }, (_, i) => (
        <div key={i} className={cx(i < lit && 'done', i === lit && 'now')} />
      ))}
    </div>
  )
}

function StageList({ active }: { active: number }) {
  return (
    <ol className="stages">
      {STAGES.map((name, i) => (
        <li key={name} className={cx(i < active && 'done', i === active && 'active')}>
          <span />
          {name}
        </li>
      ))}
    </ol>
  )
}
