import { useEffect, useState } from 'react'

const STAGES = [
  'Preprocessing channels',
  'Embedding cells',
  'Encoding tissue graph',
  'Weighing crops',
  'Attributing the decision',
]
const CUTS = [0.08, 0.55, 0.72, 0.8, 1] // stage ends, as a fraction of the expected time

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

  const active = CUTS.findIndex((c) => progress < c)
  const lit = Math.floor(progress * 21)

  return (
    <main style={{ flex: 1, display: 'grid', placeItems: 'center', padding: '40px 24px' }}>
      <div
        className="rise"
        style={{ display: 'flex', gap: 48, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}
      >
        <div
          aria-hidden
          style={{
            display: 'grid', gridTemplateColumns: 'repeat(4, 44px)', gap: 4, padding: 14,
            background: '#06080a', borderRadius: 10,
          }}
        >
          {Array.from({ length: 20 }, (_, i) => (
            <div
              key={i}
              style={{
                width: 44, height: 44, borderRadius: 2, transition: 'background .5s',
                background:
                  i < lit ? 'rgba(255,255,255,.42)' : i === lit ? 'rgba(255,255,255,.85)' : 'rgba(255,255,255,.07)',
              }}
            />
          ))}
        </div>
        <div style={{ width: 'min(100%, 340px)', display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div>
            <h1 className="serif" style={{ fontWeight: 400, fontSize: 32, lineHeight: 1.15, margin: '0 0 6px' }}>
              Reading the tissue
            </h1>
            <div className="mut" style={{ fontSize: 14, overflowWrap: 'anywhere' }}>
              {fileName} · usually about {expectedSeconds} seconds
            </div>
          </div>
          <div
            role="progressbar"
            aria-valuenow={Math.round(progress * 100)}
            style={{ height: 3, background: 'var(--line)', borderRadius: 2, overflow: 'hidden' }}
          >
            <div
              style={{ height: '100%', width: `${Math.round(progress * 100)}%`, background: 'var(--ink)', transition: 'width .2s linear' }}
            />
          </div>
          <ol style={{ margin: 0, padding: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 9 }}>
            {STAGES.map((name, i) => (
              <li
                key={name}
                style={{
                  display: 'flex', alignItems: 'center', gap: 10, fontSize: 14, transition: 'color .4s',
                  color: i <= active ? 'var(--ink)' : 'var(--mut)',
                }}
              >
                <span
                  style={{
                    width: 8, height: 8, borderRadius: '50%', transition: 'background .4s',
                    background: i < active ? 'var(--ink)' : i === active ? 'var(--a)' : 'var(--line)',
                    animation: i === active ? 'pulse 1.2s infinite' : 'none',
                  }}
                />
                {name}
              </li>
            ))}
          </ol>
          <div className="mono mut" style={{ fontSize: 12 }}>
            Stages are an estimate. The server does not report progress.
          </div>
        </div>
      </div>
    </main>
  )
}
