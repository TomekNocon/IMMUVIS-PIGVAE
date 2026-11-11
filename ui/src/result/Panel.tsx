// Side panel of the Result screen: prediction, direction tally, scatter, top crops, details.
import { useState } from 'react'
import { CLASS_VAR, igClass, mix, pct, signed, type CropView, type ResultModel } from './model'

interface Props {
  m: ResultModel
  hover: number | null
  pin: number | null
  setHover: (i: number | null) => void
  togglePin: (i: number) => void
  onAgain: () => void
}

export function Panel({ m, hover, pin, setHover, togglePin, onAgain }: Props) {
  const [sort, setSort] = useState<'att' | 'ig'>('att')
  const [details, setDetails] = useState(false)
  const cur = hover ?? pin
  const key = sort === 'att' ? (c: CropView) => c.att : (c: CropView) => Math.abs(c.ig)
  const sorted = [...m.crops].sort((x, y) => key(y) - key(x))
  const kmax = Math.max(key(sorted[0]), 1e-12)
  const n = m.crops.length
  const nA = m.crops.filter((c) => c.ig < 0).length
  const enter = (i: number) => () => setHover(i)
  const leave = () => setHover(null)

  return (
    <aside style={{ display: 'flex', flexDirection: 'column', gap: 28, minWidth: 0, animation: 'rise .5s .12s both' }}>
      <div>
        <div className="caps">Predicted subtype</div>
        <h1
          className="serif"
          style={{
            fontWeight: 400, fontSize: 'clamp(38px, 4.6vw, 58px)', lineHeight: 1.05, margin: '6px 0 16px',
            color: CLASS_VAR[m.predIdx], textWrap: 'balance',
          }}
        >
          {m.classes[m.predIdx]}
        </h1>
        <div
          style={{
            display: 'flex', height: 10, borderRadius: 2, overflow: 'hidden', gap: 2, transformOrigin: 'left',
            animation: 'grow .9s .3s cubic-bezier(.2,.7,.2,1) both',
          }}
        >
          <div style={{ width: pct(m.probs[0]), background: 'var(--a)', minWidth: 3 }} />
          <div style={{ flex: 1, background: 'var(--s)', minWidth: 3 }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, marginTop: 8, fontSize: 14 }}>
          <span>
            {m.classes[0]} <strong className="mono" style={{ fontWeight: 500 }}>{pct(m.probs[0])}</strong>
          </span>
          <span style={{ textAlign: 'right' }}>
            {m.classes[1]} <strong className="mono" style={{ fontWeight: 500 }}>{pct(m.probs[1])}</strong>
          </span>
        </div>
      </div>

      <div>
        <div className="caps" style={{ marginBottom: 8 }}>Crops by direction</div>
        <div style={{ display: 'flex', gap: 3 }} onMouseLeave={leave}>
          {[...m.crops]
            .sort((x, y) => x.ig - y.ig)
            .map((c) => (
              <div
                key={c.i}
                onMouseEnter={enter(c.i)}
                onClick={() => togglePin(c.i)}
                style={{
                  flex: 1, height: 22, cursor: 'pointer', outlineOffset: 1,
                  background: mix(CLASS_VAR[igClass(c.ig)], 0.25 + (0.75 * Math.abs(c.ig)) / m.maxIg),
                  outline: c.i === cur ? '2px solid var(--ink)' : 'none',
                }}
              />
            ))}
        </div>
        <div className="mut" style={{ marginTop: 8, fontSize: 14 }}>
          {nA} of {n} crops push toward {m.classes[0]}, {n - nA} toward {m.classes[1]}.
        </div>

        <div className="caps" style={{ margin: '28px 0 8px' }}>Attention vs attribution</div>
        <div
          onMouseLeave={leave}
          style={{ position: 'relative', height: 170, borderLeft: '1px solid var(--line)', borderBottom: '1px solid var(--line)' }}
        >
          <div style={{ position: 'absolute', left: 0, right: 0, top: '50%', borderTop: '1px dashed var(--line)' }} />
          <span className="mono" style={{ position: 'absolute', left: 8, top: 4, fontSize: 11, color: 'var(--s)' }}>
            IG → {m.short[1]}
          </span>
          <span className="mono" style={{ position: 'absolute', left: 8, bottom: 4, fontSize: 11, color: 'var(--a)' }}>
            IG → {m.short[0]}
          </span>
          <div style={{ position: 'absolute', inset: '14px 16px 14px 12px' }}>
            {m.crops.map((c) => {
              const size = c.i === cur ? 14 : 9
              return (
                <div
                  key={c.i}
                  onMouseEnter={enter(c.i)}
                  onClick={() => togglePin(c.i)}
                  style={{
                    position: 'absolute', left: `${Math.sqrt(c.att / m.maxAtt) * 100}%`,
                    top: `${50 - (c.ig / m.maxIg) * 50}%`, width: size, height: size, borderRadius: '50%',
                    background: CLASS_VAR[igClass(c.ig)], transform: 'translate(-50%, -50%)', cursor: 'pointer',
                    outline: c.i === cur ? '2px solid var(--ink)' : c.i === pin ? '2px dashed var(--ink)' : 'none',
                    outlineOffset: 2, transition: 'left .5s, top .5s, width .2s, height .2s',
                  }}
                />
              )
            })}
          </div>
        </div>
        <div className="mono mut" style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 11 }}>
          <span>low attention</span>
          <span>high attention (√ scale)</span>
        </div>
      </div>

      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 8, flexWrap: 'wrap' }}>
          <span className="caps">Top 5 crops</span>
          <div style={{ display: 'flex', gap: 12, fontSize: 13 }}>
            {(
              [
                ['att', 'by attention'],
                ['ig', 'by |IG|'],
              ] as const
            ).map(([k, label]) => (
              <button
                key={k}
                onClick={() => setSort(k)}
                aria-pressed={sort === k}
                style={{
                  font: 'inherit', background: 'none', border: 'none', padding: '0 0 2px', cursor: 'pointer',
                  color: sort === k ? 'var(--ink)' : 'var(--mut)',
                  borderBottom: `1px solid ${sort === k ? 'var(--ink)' : 'transparent'}`,
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <ol onMouseLeave={leave} style={{ margin: 0, padding: 0, listStyle: 'none', borderTop: '1px solid var(--line)' }}>
          {sorted.slice(0, 5).map((c, rank) => (
            <li
              key={c.i}
              className="toprow"
              onMouseEnter={enter(c.i)}
              onClick={() => togglePin(c.i)}
              style={{ background: c.i === cur ? 'var(--panel)' : 'transparent' }}
            >
              <span className="mut">{rank + 1}</span>
              <span style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 0 }}>
                <span>
                  row {c.row}, col {c.col}
                </span>
                <span
                  style={{
                    height: 3, width: `${(key(c) / kmax) * 100}%`, transition: 'width .5s',
                    background: sort === 'att' ? 'var(--ink)' : CLASS_VAR[igClass(c.ig)],
                  }}
                />
              </span>
              <span style={{ textAlign: 'right' }}>{pct(c.att)}</span>
              <span style={{ textAlign: 'right', color: CLASS_VAR[igClass(c.ig)] }}>{signed(c.ig)}</span>
            </li>
          ))}
        </ol>
        <div
          className="mut"
          style={{ display: 'grid', gridTemplateColumns: '22px minmax(0,1fr) 64px 64px', gap: 10, padding: '6px 8px', fontSize: 12 }}
        >
          <span />
          <span />
          <span style={{ textAlign: 'right' }}>attention</span>
          <span style={{ textAlign: 'right' }}>IG</span>
        </div>
      </div>

      <Details m={m} open={details} onToggle={() => setDetails((d) => !d)} />

      <button className="outline" style={{ alignSelf: 'flex-start' }} onClick={onAgain}>
        Analyse another image
      </button>
    </aside>
  )
}

const STAGE_COLOR = ['var(--mut)', 'var(--ink)', 'var(--mut)', 'var(--line)']

function Details({ m, open, onToggle }: { m: ResultModel; open: boolean; onToggle: () => void }) {
  const shown = m.timing.filter(([, ms]) => ms > 0)
  return (
    <div>
      <button className="textbtn" onClick={onToggle} aria-expanded={open}>
        {open ? 'Hide details' : 'Model and timing details'}
      </button>
      <div className={`expand${open ? ' open' : ''}`}>
        <div>
          <dl
            className="mono"
            style={{ margin: '12px 0 0', display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '6px 20px', fontSize: 13 }}
          >
            <dt className="mut">logit</dt>
            <dd style={{ margin: 0 }}>{m.logit}</dd>
            <dt className="mut">timing</dt>
            <dd style={{ margin: 0 }}>{(m.totalMs / 1000).toFixed(1)} s total (server side)</dd>
          </dl>
          <div style={{ display: 'flex', height: 6, gap: 2, marginTop: 10 }}>
            {shown.map(([name, ms], i) => (
              <div
                key={name}
                title={`${name} ${(ms / 1000).toFixed(1)} s`}
                style={{ flex: ms, background: STAGE_COLOR[i % STAGE_COLOR.length] }}
              />
            ))}
          </div>
          <div className="mono mut" style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginTop: 6, fontSize: 11 }}>
            {m.timing.map(([name, ms]) => (
              <span key={name}>
                {name} {(ms / 1000).toFixed(1)}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
