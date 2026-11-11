// Side panel of the Result screen: prediction, direction tally, scatter, top crops, details.
// Inline styles here are only the data-driven ones (sizes, positions, class colours).
import { useState, type ReactNode } from 'react'
import { cx } from '@/components/cx'
import { CLASS_VAR, igClass, mix, pct, signed, type CropSelection, type CropView, type ResultModel } from './model'
import './Panel.css'

type ChartProps = { m: ResultModel } & CropSelection

interface Props extends ChartProps {
  onAgain: () => void
  /** The "Ask about this result" bar, shown under the prediction. */
  ask?: ReactNode
}

export function Panel({ ask, onAgain, ...chart }: Props) {
  return (
    <aside className="sidePanel">
      <Prediction m={chart.m} />
      {ask}
      <div>
        <DirectionStrip {...chart} />
        <AttentionScatter {...chart} />
      </div>
      <TopCrops {...chart} />
      <Details m={chart.m} />
      <button className="outline alignStart" onClick={onAgain}>
        Analyse another image
      </button>
    </aside>
  )
}

function Prediction({ m }: { m: ResultModel }) {
  return (
    <div>
      <div className="caps">Predicted subtype</div>
      <h1 className="title predTitle" style={{ color: CLASS_VAR[m.predIdx] }}>
        {m.classes[m.predIdx]}
      </h1>
      <div className="probBar">
        <div style={{ width: pct(m.probs[0]) }} />
        <div />
      </div>
      <div className="probLabels">
        {[0, 1].map((k) => (
          <span key={k}>
            {m.classes[k]} <strong className="mono">{pct(m.probs[k])}</strong>
          </span>
        ))}
      </div>
    </div>
  )
}

/** One cell per crop, ordered from most toward classes[0] to most toward classes[1]. */
function DirectionStrip({ m, hover, pin, setHover, togglePin }: ChartProps) {
  const cur = hover ?? pin
  const nA = m.crops.filter((c) => c.ig < 0).length
  return (
    <>
      <div className="caps head">Crops by direction</div>
      <div className="dirStrip" onMouseLeave={() => setHover(null)}>
        {[...m.crops]
          .sort((x, y) => x.ig - y.ig)
          .map((c) => (
            <div
              key={c.i}
              onMouseEnter={() => setHover(c.i)}
              onClick={() => togglePin(c.i)}
              style={{
                background: mix(CLASS_VAR[igClass(c.ig)], 0.25 + (0.75 * Math.abs(c.ig)) / m.maxIg),
                outline: c.i === cur ? '2px solid var(--ink)' : 'none',
              }}
            />
          ))}
      </div>
      <div className="mut dirNote">
        {nA} of {m.crops.length} crops push toward {m.classes[0]}, {m.crops.length - nA} toward {m.classes[1]}.
      </div>
    </>
  )
}

/** Attention (x, square-root scale) against IG (y, signed). */
function AttentionScatter({ m, hover, pin, setHover, togglePin }: ChartProps) {
  const cur = hover ?? pin
  return (
    <>
      <div className="caps head scatterTitle">Attention vs attribution</div>
      <div className="scatter" onMouseLeave={() => setHover(null)}>
        <div className="scatterZero" />
        <span className="mono scatterAxis toS">IG → {m.short[1]}</span>
        <span className="mono scatterAxis toA">IG → {m.short[0]}</span>
        <div className="scatterDots">
          {m.crops.map((c) => {
            const size = c.i === cur ? 14 : 9
            return (
              <div
                key={c.i}
                onMouseEnter={() => setHover(c.i)}
                onClick={() => togglePin(c.i)}
                style={{
                  left: `${Math.sqrt(c.att / m.maxAtt) * 100}%`,
                  top: `${50 - (c.ig / m.maxIg) * 50}%`,
                  width: size,
                  height: size,
                  background: CLASS_VAR[igClass(c.ig)],
                  outline: c.i === cur ? '2px solid var(--ink)' : c.i === pin ? '2px dashed var(--ink)' : 'none',
                }}
              />
            )
          })}
        </div>
      </div>
      <div className="mono mut scatterScale">
        <span>low attention</span>
        <span>high attention (√ scale)</span>
      </div>
    </>
  )
}

const SORTS = [
  ['att', 'by attention'],
  ['ig', 'by |IG|'],
] as const

function TopCrops({ m, hover, pin, setHover, togglePin }: ChartProps) {
  const [sort, setSort] = useState<'att' | 'ig'>('att')
  const cur = hover ?? pin
  const key = sort === 'att' ? (c: CropView) => c.att : (c: CropView) => Math.abs(c.ig)
  const sorted = [...m.crops].sort((x, y) => key(y) - key(x))
  const kmax = Math.max(key(sorted[0]), 1e-12)

  return (
    <div>
      <div className="topHead">
        <span className="caps">Top 5 crops</span>
        <div className="sortTabs">
          {SORTS.map(([k, label]) => (
            <button key={k} className={cx('sortTab', sort === k && 'on')} onClick={() => setSort(k)} aria-pressed={sort === k}>
              {label}
            </button>
          ))}
        </div>
      </div>
      <ol className="topList" onMouseLeave={() => setHover(null)}>
        {sorted.slice(0, 5).map((c, rank) => (
          <li
            key={c.i}
            className={cx('toprow', c.i === cur && 'on')}
            onMouseEnter={() => setHover(c.i)}
            onClick={() => togglePin(c.i)}
          >
            <span className="mut">{rank + 1}</span>
            <span className="topCrop">
              <span>
                row {c.row}, col {c.col}
              </span>
              <span
                className="topBar"
                style={{
                  width: `${(key(c) / kmax) * 100}%`,
                  background: sort === 'att' ? 'var(--ink)' : CLASS_VAR[igClass(c.ig)],
                }}
              />
            </span>
            <span className="num">{pct(c.att)}</span>
            <span className="num" style={{ color: CLASS_VAR[igClass(c.ig)] }}>
              {signed(c.ig)}
            </span>
          </li>
        ))}
      </ol>
      <div className="mut topLegend">
        <span />
        <span />
        <span className="num">attention</span>
        <span className="num">IG</span>
      </div>
    </div>
  )
}

const STAGE_COLOR = ['var(--mut)', 'var(--ink)', 'var(--mut)', 'var(--line)']

function Details({ m }: { m: ResultModel }) {
  const [open, setOpen] = useState(false)
  return (
    <div>
      <button className="textbtn" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {open ? 'Hide details' : 'Model and timing details'}
      </button>
      <div className={cx('expand', open && 'open')}>
        <div>
          <dl className="mono detailsList">
            <dt className="mut">logit</dt>
            <dd>{m.logit}</dd>
            <dt className="mut">timing</dt>
            <dd>{(m.totalMs / 1000).toFixed(1)} s total (server side)</dd>
          </dl>
          <div className="timingBar">
            {m.timing
              .filter(([, ms]) => ms > 0)
              .map(([name, ms], i) => (
                <div
                  key={name}
                  title={`${name} ${(ms / 1000).toFixed(1)} s`}
                  style={{ flex: ms, background: STAGE_COLOR[i % STAGE_COLOR.length] }}
                />
              ))}
          </div>
          <div className="mono mut timingNames">
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
