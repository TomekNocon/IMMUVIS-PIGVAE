/** The first crop an answer mentions ("row 0, col 128"), if any. */
export function firstCrop(text: string): { row: number; col: number } | null {
  const m = /row\s+(\d+),\s*col(?:umn)?\s+(\d+)/i.exec(text)
  return m ? { row: Number(m[1]), col: Number(m[2]) } : null
}
