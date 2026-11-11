// Minimal Markdown for chat replies: paragraphs, "-"/"*"/"1." lists, headings, **bold**, `code`.
// Builds React elements (no innerHTML), so a reply cannot inject markup into the page.
// Mentions like "row 0, col 128" can be turned into links to the crop (see CropRef).
import type { ReactNode } from 'react'

/** Renders a crop mention; returns null when there is no such crop (left as plain text). */
export type CropRef = (row: number, col: number, label: string, key: string) => ReactNode | null

const CROP = /(row\s+\d+,\s*col(?:umn)?\s+\d+)/gi

function crops(text: string, cropRef: CropRef | undefined, keyBase: string): ReactNode[] {
  if (!cropRef) return [text]
  return text.split(CROP).map((part, i) => {
    const m = /row\s+(\d+),\s*col(?:umn)?\s+(\d+)/i.exec(part)
    if (!m || i % 2 === 0) return part
    return cropRef(Number(m[1]), Number(m[2]), part, `${keyBase}-${i}`) ?? part
  })
}

function inline(text: string, cropRef?: CropRef): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return <strong key={i}>{crops(part.slice(2, -2), cropRef, `b${i}`)}</strong>
    }
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return (
        <code key={i} className="mono" style={{ fontSize: '0.9em' }}>
          {part.slice(1, -1)}
        </code>
      )
    }
    return <span key={i}>{crops(part, cropRef, `t${i}`)}</span>
  })
}

const LIST_ITEM = /^\s*(?:[-*]|\d+[.)])\s+/

export function Markdown({ text, cropRef }: { text: string; cropRef?: CropRef }) {
  const blocks = text.trim().split(/\n\s*\n/)
  return (
    <>
      {blocks.map((block, b) => {
        const lines = block.split('\n').filter((l) => l.trim())
        if (lines.length && lines.every((l) => LIST_ITEM.test(l))) {
          const ordered = /^\s*\d/.test(lines[0])
          const items = lines.map((l, i) => <li key={i}>{inline(l.replace(LIST_ITEM, ''), cropRef)}</li>)
          return ordered ? <ol key={b}>{items}</ol> : <ul key={b}>{items}</ul>
        }
        if (lines.length === 1 && /^#{1,6}\s/.test(lines[0])) {
          return (
            <p key={b}>
              <strong>{inline(lines[0].replace(/^#{1,6}\s+/, ''), cropRef)}</strong>
            </p>
          )
        }
        return (
          <p key={b}>
            {lines.map((l, i) => (
              <span key={i}>
                {i > 0 && <br />}
                {inline(l.replace(LIST_ITEM, (m) => (/\d/.test(m) ? m : '• ')), cropRef)}
              </span>
            ))}
          </p>
        )
      })}
    </>
  )
}
