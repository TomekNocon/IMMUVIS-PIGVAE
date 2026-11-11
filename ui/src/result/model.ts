// Derived view data for the Result screen, computed once per prediction.
import type { PredictResponse } from '@/api'

export interface CropView {
  i: number
  row: number
  col: number
  att: number
  ig: number
  attRank: number // 0 = highest attention
  igRank: number // 0 = largest |IG|
}

/** The hovered and pinned crop (indices into ResultModel.crops), shared by the viewer and the panel. */
export interface CropSelection {
  hover: number | null
  pin: number | null
  setHover: (i: number | null) => void
  togglePin: (i: number) => void
}

export interface ResultModel {
  crops: CropView[]
  maxAtt: number
  maxIg: number
  classes: [string, string] // [classes[0] -> --a, classes[1] -> --s]
  short: [string, string] // first word of each class, for compact labels
  predIdx: 0 | 1
  probs: [number, number]
  logit: string
  width: number // original image pixels
  height: number
  tile: number
  coveredW: number // pixels covered by whole crops; the rest was not analysed
  coveredH: number
  previewSrc: string
  previewMarkers: string[]
  timing: [string, number][] // stage -> ms
  totalMs: number
}

export function buildModel(r: PredictResponse): ResultModel {
  const tile = r.image.tile_size
  const crops: CropView[] = r.crops.map((c, i) => ({
    i,
    row: c.row,
    col: c.col,
    att: c.attention,
    ig: c.ig_importance ?? 0,
    attRank: 0,
    igRank: 0,
  }))
  ;[...crops].sort((x, y) => y.att - x.att).forEach((c, n) => (c.attRank = n))
  ;[...crops].sort((x, y) => Math.abs(y.ig) - Math.abs(x.ig)).forEach((c, n) => (c.igRank = n))

  const names = Object.keys(r.probabilities)
  const classes: [string, string] = [names[0] ?? 'class 0', names[1] ?? 'class 1']
  const probs: [number, number] = [r.probabilities[classes[0]] ?? 0, r.probabilities[classes[1]] ?? 0]
  const timing = Object.entries(r.timing_ms)
  const logit = Array.isArray(r.logit) ? r.logit.map((x) => x.toFixed(3)).join(', ') : signed(r.logit)

  return {
    crops,
    maxAtt: Math.max(...crops.map((c) => c.att), 1e-12),
    maxIg: Math.max(...crops.map((c) => Math.abs(c.ig)), 1e-12),
    classes,
    short: [classes[0].split(' ')[0], classes[1].split(' ')[0]],
    predIdx: r.prediction === classes[1] ? 1 : 0,
    probs,
    logit,
    width: r.image.width,
    height: r.image.height,
    tile,
    coveredW: Math.max(...crops.map((c) => c.col)) + tile,
    coveredH: Math.max(...crops.map((c) => c.row)) + tile,
    previewSrc: `data:image/png;base64,${r.preview.png_base64}`,
    previewMarkers: r.preview.markers,
    timing,
    totalMs: timing.reduce((a, [, ms]) => a + ms, 0),
  }
}

/** "+0.123" / "−0.123" with a real minus sign, as in the design. */
export function signed(x: number, digits = 3): string {
  return (x > 0 ? '+' : '−') + Math.abs(x).toFixed(digits)
}

export const pct = (x: number) => `${(x * 100).toFixed(1)}%`

// Class colours: themed (side panel) and fixed (inside the always-dark viewer).
export const CLASS_VAR = ['var(--a)', 'var(--s)'] as const
export const CLASS_FIXED = ['oklch(0.76 0.13 245)', 'oklch(0.78 0.13 60)'] as const
export const ATT_HEAT = 'oklch(0.9 0.17 100)'

/** IG > 0 pushes toward classes[1]. */
export const igClass = (ig: number): 0 | 1 => (ig > 0 ? 1 : 0)

export const mix = (color: string, alpha: number) =>
  `color-mix(in oklch, ${color} ${Math.round(alpha * 100)}%, transparent)`
