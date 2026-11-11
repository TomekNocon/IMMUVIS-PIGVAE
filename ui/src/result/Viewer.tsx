// The always-dark image viewer: preview image, crop heatmap, zoom/pan, tooltip and pin/compare.
import { useEffect, useRef, useState, type MouseEvent, type PointerEvent } from 'react'
import { ATT_HEAT, CLASS_FIXED, igClass, mix, pct, signed, type CropView, type ResultModel } from './model'

type Heat = 'att' | 'ig' | 'off'

interface Props {
  m: ResultModel
  hover: number | null
  pin: number | null
  setHover: (i: number | null) => void
  togglePin: (i: number) => void
}

interface View {
  z: number
  tx: number // pan, as a fraction of the stage size
  ty: number
}

const clampView = (z: number, tx: number, ty: number): View => ({
  z,
  tx: Math.min(0, Math.max(1 - z, tx)),
  ty: Math.min(0, Math.max(1 - z, ty)),
})

export function Viewer({ m, hover, pin, setHover, togglePin }: Props) {
  const [heat, setHeat] = useState<Heat>('att')
  const [opacity, setOpacity] = useState(0.7)
  const [view, setView] = useState<View>({ z: 1, tx: 0, ty: 0 })
  const [dragging, setDragging] = useState(false)
  const stageRef = useRef<HTMLDivElement>(null)
  const dragStart = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null)
  const moved = useRef(false) // a drag of more than 4px is not a click
  const viewRef = useRef(view)
  viewRef.current = view

  const W = m.width
  const H = m.height
  const cur = hover ?? pin
  const tc = cur != null ? m.crops[cur] : null

  const zoomAt = (z2: number, fx: number, fy: number) => {
    const { z, tx, ty } = viewRef.current
    const nz = Math.min(4, Math.max(1, z2))
    setView(clampView(nz, fx - ((fx - tx) * nz) / z, fy - ((fy - ty) * nz) / z))
  }
  const frac = (e: { clientX: number; clientY: number }) => {
    const r = stageRef.current!.getBoundingClientRect()
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height] as const
  }

  // Ctrl/⌘ + wheel (and trackpad pinch) zoom; needs a non-passive listener to prevent page zoom.
  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return
      e.preventDefault()
      const [fx, fy] = frac(e)
      zoomAt(viewRef.current.z * (e.deltaY < 0 ? 1.15 : 1 / 1.15), fx, fy)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const onDown = (e: PointerEvent) => {
    dragStart.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty }
    moved.current = false
  }
  const onMove = (e: PointerEvent) => {
    const d = dragStart.current
    if (!d || view.z === 1) return
    const r = stageRef.current!.getBoundingClientRect()
    const dx = e.clientX - d.x
    const dy = e.clientY - d.y
    if (Math.abs(dx) + Math.abs(dy) > 4) moved.current = true
    if (moved.current) {
      setDragging(true)
      setHover(null)
      setView(clampView(view.z, d.tx + dx / r.width, d.ty + dy / r.height))
    }
  }
  const onUp = () => {
    dragStart.current = null
    setDragging(false)
    setTimeout(() => (moved.current = false), 0)
  }
  const onDouble = (e: MouseEvent) => {
    const [fx, fy] = frac(e)
    if (view.z >= 4) setView({ z: 1, tx: 0, ty: 0 })
    else zoomAt(view.z * 2, fx, fy)
  }

  const fill = (c: CropView) => {
    if (heat === 'att') return `oklch(0.9 0.17 100 / ${(c.att / m.maxAtt).toFixed(3)})`
    if (heat === 'ig') return mix(CLASS_FIXED[igClass(c.ig)], Math.abs(c.ig) / m.maxIg)
    return 'transparent'
  }

  // Tooltip beside the crop, flipped to stay inside the stage.
  const tipPos = (c: CropView) => {
    const { z, tx, ty } = view
    const x0 = (c.col / W) * z + tx
    const x1 = ((c.col + m.tile) / W) * z + tx
    const y0 = (c.row / H) * z + ty
    const y1 = ((c.row + m.tile) / H) * z + ty
    const right = (x0 + x1) / 2 < 0.5
    const below = (y0 + y1) / 2 < 0.5
    return {
      left: `${(right ? Math.min(x1, 0.55) : Math.max(x0, 0.45)) * 100}%`,
      top: `${(below ? Math.max(y0, 0) : Math.min(y1, 1)) * 100}%`,
      transform: `translate(${right ? '8px' : 'calc(-100% - 8px)'}, ${below ? '0' : '-100%'})`,
    }
  }

  const rightFrac = 1 - m.coveredW / W
  const bottomFrac = 1 - m.coveredH / H
  const pc = pin != null ? m.crops[pin] : null
  const hc = pc && hover != null && hover !== pin ? m.crops[hover] : null

  return (
    <section className="viewer">
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        <div className="track" role="group" aria-label="Heatmap">
          {(
            [
              ['att', 'Attention'],
              ['ig', 'Integrated Gradients'],
              ['off', 'Off'],
            ] as const
          ).map(([k, label]) => (
            <button key={k} className={`seg${heat === k ? ' on' : ''}`} aria-pressed={heat === k} onClick={() => setHeat(k)}>
              {label}
            </button>
          ))}
        </div>
        <label
          style={{
            display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: '#9aa4ae',
            opacity: heat === 'off' ? 0.35 : 1, transition: 'opacity .3s',
          }}
        >
          Opacity
          <input
            type="range"
            min={0.1}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(+e.target.value)}
            style={{ width: 110, accentColor: '#eef0f2' }}
          />
        </label>
        <div className="track" style={{ marginLeft: 'auto', alignItems: 'center' }}>
          <button className="vbtn" aria-label="Zoom out" onClick={() => zoomAt(view.z / 1.5, 0.5, 0.5)}>
            −
          </button>
          <button
            className="vbtn mono"
            title="Reset zoom"
            style={{ width: 52, fontSize: 12, color: '#c9d0d6' }}
            onClick={() => setView({ z: 1, tx: 0, ty: 0 })}
          >
            {Math.round(view.z * 100)}%
          </button>
          <button className="vbtn" aria-label="Zoom in" onClick={() => zoomAt(view.z * 1.5, 0.5, 0.5)}>
            +
          </button>
        </div>
      </div>

      <div
        ref={stageRef}
        onMouseLeave={() => setHover(null)}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={onUp}
        onDoubleClick={onDouble}
        style={{
          cursor: view.z > 1 ? (dragging ? 'grabbing' : 'grab') : 'default',
          userSelect: 'none', position: 'relative', alignSelf: 'center',
          width: `min(100%, calc(74vh * ${W} / ${H}))`, aspectRatio: `${W} / ${H}`,
          overflow: 'hidden', borderRadius: 4, touchAction: view.z > 1 ? 'none' : 'auto',
        }}
      >
        <div
          style={{
            position: 'absolute', inset: 0, transformOrigin: '0 0',
            transform: `translate(${view.tx * 100}%, ${view.ty * 100}%) scale(${view.z})`,
            transition: dragging ? 'none' : 'transform .25s ease',
          }}
        >
          <img
            src={m.previewSrc}
            alt={`IMC preview, ${m.previewMarkers.join(' / ')} as red / green / blue`}
            draggable={false}
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', maxWidth: 'none' }}
          />
          {m.crops.map((c) => (
            <div
              key={c.i}
              onMouseEnter={() => setHover(c.i)}
              onClick={() => !moved.current && togglePin(c.i)}
              style={{
                position: 'absolute', left: `${(c.col / W) * 100}%`, top: `${(c.row / H) * 100}%`,
                width: `${(m.tile / W) * 100}%`, height: `${(m.tile / H) * 100}%`, cursor: 'pointer',
                animation: 'cropIn .6s both', animationDelay: `${c.attRank * 45}ms`,
              }}
            >
              <div style={{ position: 'absolute', inset: 0, background: fill(c), opacity, transition: 'background .45s, opacity .3s' }} />
              <div
                style={{
                  position: 'absolute', inset: 0, transition: 'border-color .2s',
                  border: c.i === cur ? '2px solid #fff' : c.i === pin ? '2px dashed #fff' : '1px solid rgba(255,255,255,.16)',
                }}
              />
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
        {tc && (
          <div className="tip" style={tipPos(tc)}>
            <div style={{ color: '#9aa4ae' }}>
              row {tc.row}, col {tc.col}
            </div>
            <div>
              attention {pct(tc.att)} <span style={{ color: '#9aa4ae' }}>#{tc.attRank + 1}</span>
            </div>
            <div>
              IG <span style={{ color: CLASS_FIXED[igClass(tc.ig)] }}>{signed(tc.ig)}</span>{' '}
              <span style={{ color: '#9aa4ae' }}>#{tc.igRank + 1}</span>
            </div>
            <div style={{ color: CLASS_FIXED[igClass(tc.ig)] }}>toward {m.short[igClass(tc.ig)]}</div>
          </div>
        )}
      </div>

      <div className="mono" style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 11, color: '#c9d0d6', minHeight: 18, flexWrap: 'wrap' }}>
        <span>{heat === 'att' ? '0' : heat === 'ig' ? `← ${m.short[0]}` : ''}</span>
        <span
          style={{
            flex: 1, minWidth: 80, height: 8, borderRadius: 2, transition: 'opacity .3s', opacity: heat === 'off' ? 0 : 1,
            background:
              heat === 'ig'
                ? `linear-gradient(90deg, ${CLASS_FIXED[0]}, transparent 50%, ${CLASS_FIXED[1]})`
                : `linear-gradient(90deg, transparent, ${ATT_HEAT})`,
          }}
        />
        <span>{heat === 'att' ? `max attention ${pct(m.maxAtt)}` : heat === 'ig' ? `${m.short[1]} →` : ''}</span>
      </div>
      <div className="mono" style={{ fontSize: 11, color: '#9aa4ae' }}>
        {m.previewMarkers.join(' / ')} as red / green / blue · {W} × {H} px · {m.crops.length} crops of {m.tile} px.
        Double-click or Ctrl/⌘ + scroll to zoom, drag to pan.
      </div>

      <div
        className="mono"
        style={{ minHeight: 66, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 10, fontSize: 12, lineHeight: 1.6 }}
      >
        {!pc && (
          <div style={{ display: 'flex', alignItems: 'center', padding: '10px 12px', border: '1px dashed #3a424b', borderRadius: 6, color: '#9aa4ae' }}>
            Click a crop to pin it, then hover another to compare.
          </div>
        )}
        {pc && (
          <div style={{ padding: '10px 12px', border: '1px dashed #c9d0d6', borderRadius: 6, animation: 'rise .3s both' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, color: '#9aa4ae' }}>
              <span>
                PINNED · row {pc.row}, col {pc.col}
              </span>
              <button className="vlink" onClick={() => togglePin(pc.i)}>
                unpin
              </button>
            </div>
            <div>
              attention {pct(pc.att)} · IG <span style={{ color: CLASS_FIXED[igClass(pc.ig)] }}>{signed(pc.ig)}</span>
            </div>
          </div>
        )}
        {pc && hc && (
          <div style={{ padding: '10px 12px', border: '1px solid #c9d0d6', borderRadius: 6 }}>
            <div style={{ color: '#9aa4ae' }}>
              HOVER · row {hc.row}, col {hc.col}
            </div>
            <div>
              attention {pct(hc.att)} · IG <span style={{ color: CLASS_FIXED[igClass(hc.ig)] }}>{signed(hc.ig)}</span>
            </div>
            <div style={{ color: '#9aa4ae' }}>
              Δ attention {signed((hc.att - pc.att) * 100, 1)} pts · Δ IG {signed(hc.ig - pc.ig)}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
