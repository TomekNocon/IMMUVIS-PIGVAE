// The always-dark image viewer: preview image, crop heatmap, zoom/pan, tooltip and pin/compare.
// Inline styles here are only the data-driven ones (crop positions, heat colours, zoom transform).
import { useState } from 'react'
import { cx } from '@/components/cx'
import { ATT_HEAT, CLASS_FIXED, igClass, mix, pct, signed, type CropSelection, type CropView, type ResultModel } from './model'
import { useZoomPan, type View } from './useZoomPan'
import './Viewer.css'

type Heat = 'att' | 'ig' | 'off'

const HEATS = [
  ['att', 'Attention'],
  ['ig', 'Integrated Gradients'],
  ['off', 'Off'],
] as const

export function Viewer({ m, hover, pin, setHover, togglePin }: { m: ResultModel } & CropSelection) {
  const [heat, setHeat] = useState<Heat>('att')
  const [opacity, setOpacity] = useState(0.7)
  const { stageRef, view, dragging, handlers, zoomBy, reset, wasDrag } = useZoomPan(() => setHover(null))

  const W = m.width
  const H = m.height
  const cur = hover ?? pin
  const fill = (c: CropView) => {
    if (heat === 'att') return `oklch(0.9 0.17 100 / ${(c.att / m.maxAtt).toFixed(3)})`
    if (heat === 'ig') return mix(CLASS_FIXED[igClass(c.ig)], Math.abs(c.ig) / m.maxIg)
    return 'transparent'
  }
  // Strips at the right and bottom that no whole crop covers.
  const rightFrac = 1 - m.coveredW / W
  const bottomFrac = 1 - m.coveredH / H

  return (
    <section className="viewer">
      <div className="viewerBar">
        <div className="track" role="group" aria-label="Heatmap">
          {HEATS.map(([k, label]) => (
            <button key={k} className={cx('seg', heat === k && 'on')} aria-pressed={heat === k} onClick={() => setHeat(k)}>
              {label}
            </button>
          ))}
        </div>
        <label className={cx('opacityCtl', heat === 'off' && 'off')}>
          Opacity
          <input type="range" min={0.1} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} />
        </label>
        <div className="track zoomCtl">
          <button className="vbtn" aria-label="Zoom out" onClick={() => zoomBy(1 / 1.5)}>
            −
          </button>
          <button className="vbtn mono zoomLevel" title="Reset zoom" onClick={reset}>
            {Math.round(view.z * 100)}%
          </button>
          <button className="vbtn" aria-label="Zoom in" onClick={() => zoomBy(1.5)}>
            +
          </button>
        </div>
      </div>

      <div
        ref={stageRef}
        className={cx('stage', view.z > 1 && 'zoomed', dragging && 'dragging')}
        onMouseLeave={() => setHover(null)}
        {...handlers}
        style={{ width: `min(100%, calc(74vh * ${W} / ${H}))`, aspectRatio: `${W} / ${H}` }}
      >
        <div className="stageContent" style={{ transform: `translate(${view.tx * 100}%, ${view.ty * 100}%) scale(${view.z})` }}>
          <img
            src={m.previewSrc}
            alt={`IMC preview, ${m.previewMarkers.join(' / ')} as red / green / blue`}
            draggable={false}
          />
          {m.crops.map((c) => (
            <div
              key={c.i}
              className="crop"
              onMouseEnter={() => setHover(c.i)}
              onClick={() => !wasDrag() && togglePin(c.i)}
              style={{
                left: `${(c.col / W) * 100}%`,
                top: `${(c.row / H) * 100}%`,
                width: `${(m.tile / W) * 100}%`,
                height: `${(m.tile / H) * 100}%`,
                animationDelay: `${c.attRank * 45}ms`,
              }}
            >
              <div className="cropFill" style={{ background: fill(c), opacity }} />
              <div className={cx('cropEdge', c.i === cur ? 'current' : c.i === pin && 'pinned')} />
            </div>
          ))}
          {rightFrac > 0 && (
            <div className="hatch" style={{ left: `${(m.coveredW / W) * 100}%`, top: 0, right: 0, bottom: 0 }}>
              {bottomFrac < rightFrac && rightFrac > 0.12 ? 'not analysed' : null}
            </div>
          )}
          {bottomFrac > 0 && (
            <div className="hatch" style={{ left: 0, top: `${(m.coveredH / H) * 100}%`, width: `${(m.coveredW / W) * 100}%`, bottom: 0 }}>
              {bottomFrac >= rightFrac && bottomFrac > 0.04 ? 'not analysed' : null}
            </div>
          )}
        </div>
        {cur != null && <CropTooltip m={m} c={m.crops[cur]} view={view} />}
      </div>

      <HeatLegend m={m} heat={heat} />
      <div className="mono viewerNote">
        {m.previewMarkers.join(' / ')} as red / green / blue · {W} × {H} px · {m.crops.length} crops of {m.tile} px.
        Double-click or Ctrl/⌘ + scroll to zoom, drag to pan.
      </div>
      <PinCompare m={m} hover={hover} pin={pin} togglePin={togglePin} />
    </section>
  )
}

/** IG value coloured by the class it pushes toward. */
const Ig = ({ ig }: { ig: number }) => <span style={{ color: CLASS_FIXED[igClass(ig)] }}>{signed(ig)}</span>

/** Beside the crop, flipped to stay inside the stage. */
function CropTooltip({ m, c, view }: { m: ResultModel; c: CropView; view: View }) {
  const { z, tx, ty } = view
  const x0 = (c.col / m.width) * z + tx
  const x1 = ((c.col + m.tile) / m.width) * z + tx
  const y0 = (c.row / m.height) * z + ty
  const y1 = ((c.row + m.tile) / m.height) * z + ty
  const right = (x0 + x1) / 2 < 0.5
  const below = (y0 + y1) / 2 < 0.5
  const pos = {
    left: `${(right ? Math.min(x1, 0.55) : Math.max(x0, 0.45)) * 100}%`,
    top: `${(below ? Math.max(y0, 0) : Math.min(y1, 1)) * 100}%`,
    transform: `translate(${right ? '8px' : 'calc(-100% - 8px)'}, ${below ? '0' : '-100%'})`,
  }
  return (
    <div className="tip" style={pos}>
      <div className="vmut">
        row {c.row}, col {c.col}
      </div>
      <div>
        attention {pct(c.att)} <span className="vmut">#{c.attRank + 1}</span>
      </div>
      <div>
        IG <Ig ig={c.ig} /> <span className="vmut">#{c.igRank + 1}</span>
      </div>
      <div style={{ color: CLASS_FIXED[igClass(c.ig)] }}>toward {m.short[igClass(c.ig)]}</div>
    </div>
  )
}

function HeatLegend({ m, heat }: { m: ResultModel; heat: Heat }) {
  const ramp =
    heat === 'ig'
      ? `linear-gradient(90deg, ${CLASS_FIXED[0]}, transparent 50%, ${CLASS_FIXED[1]})`
      : `linear-gradient(90deg, transparent, ${ATT_HEAT})`
  return (
    <div className="mono heatLegend">
      <span>{heat === 'att' ? '0' : heat === 'ig' ? `← ${m.short[0]}` : ''}</span>
      <span className={cx('heatRamp', heat === 'off' && 'off')} style={{ background: ramp }} />
      <span>{heat === 'att' ? `max attention ${pct(m.maxAtt)}` : heat === 'ig' ? `${m.short[1]} →` : ''}</span>
    </div>
  )
}

/** The pinned crop, and the hovered one compared with it. */
function PinCompare({ m, hover, pin, togglePin }: { m: ResultModel } & Omit<CropSelection, 'setHover'>) {
  const pc = pin != null ? m.crops[pin] : null
  const hc = pc && hover != null && hover !== pin ? m.crops[hover] : null
  return (
    <div className="mono compare">
      {!pc && <div className="compareCard empty">Click a crop to pin it, then hover another to compare.</div>}
      {pc && (
        <div className="compareCard pinned">
          <div className="compareHead">
            <span>
              PINNED · row {pc.row}, col {pc.col}
            </span>
            <button className="vlink" onClick={() => togglePin(pc.i)}>
              unpin
            </button>
          </div>
          <div>
            attention {pct(pc.att)} · IG <Ig ig={pc.ig} />
          </div>
        </div>
      )}
      {pc && hc && (
        <div className="compareCard hovered">
          <div className="vmut">
            HOVER · row {hc.row}, col {hc.col}
          </div>
          <div>
            attention {pct(hc.att)} · IG <Ig ig={hc.ig} />
          </div>
          <div className="vmut">
            Δ attention {signed((hc.att - pc.att) * 100, 1)} pts · Δ IG {signed(hc.ig - pc.ig)}
          </div>
        </div>
      )}
    </div>
  )
}
