// Zoom (buttons, double-click, Ctrl/⌘ + wheel or pinch) and drag-to-pan for the image viewer.
import { useEffect, useRef, useState, type MouseEvent, type PointerEvent } from 'react'

export interface View {
  z: number // zoom, 1 to MAX_ZOOM
  tx: number // pan, as a fraction of the stage size
  ty: number
}

const MAX_ZOOM = 4
const HOME: View = { z: 1, tx: 0, ty: 0 }

const clampView = (z: number, tx: number, ty: number): View => ({
  z,
  tx: Math.min(0, Math.max(1 - z, tx)),
  ty: Math.min(0, Math.max(1 - z, ty)),
})

/** `onPanStart` runs when a drag starts panning (the viewer clears its hover then). */
export function useZoomPan(onPanStart: () => void) {
  const stageRef = useRef<HTMLDivElement>(null)
  const [view, setView] = useState<View>(HOME)
  const [dragging, setDragging] = useState(false)
  const dragStart = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null)
  const moved = useRef(false) // a drag of more than 4px is not a click
  const viewRef = useRef(view)
  viewRef.current = view

  /** Zoom to z2, keeping the point at stage fraction (fx, fy) in place. */
  const zoomAt = (z2: number, fx: number, fy: number) => {
    const { z, tx, ty } = viewRef.current
    const nz = Math.min(MAX_ZOOM, Math.max(1, z2))
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

  const onUp = () => {
    dragStart.current = null
    setDragging(false)
    setTimeout(() => (moved.current = false), 0)
  }

  /** Spread onto the stage element. */
  const handlers = {
    onPointerDown: (e: PointerEvent) => {
      dragStart.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty }
      moved.current = false
    },
    onPointerMove: (e: PointerEvent) => {
      const d = dragStart.current
      if (!d || view.z === 1) return
      const r = stageRef.current!.getBoundingClientRect()
      const dx = e.clientX - d.x
      const dy = e.clientY - d.y
      if (Math.abs(dx) + Math.abs(dy) > 4) moved.current = true
      if (moved.current) {
        setDragging(true)
        onPanStart()
        setView(clampView(view.z, d.tx + dx / r.width, d.ty + dy / r.height))
      }
    },
    onPointerUp: onUp,
    onPointerLeave: onUp,
    onDoubleClick: (e: MouseEvent) => {
      const [fx, fy] = frac(e)
      if (view.z >= MAX_ZOOM) setView(HOME)
      else zoomAt(view.z * 2, fx, fy)
    },
  }

  return {
    stageRef,
    view,
    dragging,
    handlers,
    /** Zoom in or out around the stage centre. */
    zoomBy: (factor: number) => zoomAt(view.z * factor, 0.5, 0.5),
    reset: () => setView(HOME),
    /** True right after a drag, so the click that ends it does not pin a crop. */
    wasDrag: () => moved.current,
  }
}
