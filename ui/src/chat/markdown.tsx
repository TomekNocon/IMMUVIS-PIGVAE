// Minimal Markdown for chat replies: paragraphs, "-"/"*"/"1." lists, headings, **bold**, `code`.
// Builds React elements (no innerHTML), so a reply cannot inject markup into the page.
import type { ReactNode } from 'react'

function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) return <strong key={i}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return (
        <code key={i} className="mono" style={{ fontSize: '0.9em' }}>
          {part.slice(1, -1)}
        </code>
      )
    }
    return part
  })
}

const LIST_ITEM = /^\s*(?:[-*]|\d+[.)])\s+/

export function Markdown({ text }: { text: string }) {
  const blocks = text.trim().split(/\n\s*\n/)
  return (
    <>
      {blocks.map((block, b) => {
        const lines = block.split('\n').filter((l) => l.trim())
        if (lines.length && lines.every((l) => LIST_ITEM.test(l))) {
          const ordered = /^\s*\d/.test(lines[0])
          const items = lines.map((l, i) => <li key={i}>{inline(l.replace(LIST_ITEM, ''))}</li>)
          return ordered ? <ol key={b}>{items}</ol> : <ul key={b}>{items}</ul>
        }
        const heading = lines.length === 1 && /^#{1,6}\s/.test(lines[0])
        if (heading) return <p key={b}><strong>{inline(lines[0].replace(/^#{1,6}\s+/, ''))}</strong></p>
        return (
          <p key={b}>
            {lines.map((l, i) => (
              <span key={i}>
                {i > 0 && <br />}
                {inline(l.replace(LIST_ITEM, (m) => (/\d/.test(m) ? m : '• ')))}
              </span>
            ))}
          </p>
        )
      })}
    </>
  )
}
