import type { ReactNode } from 'react'

/** A labelled form control: the label text above the input. */
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="field">
      {label}
      {children}
    </label>
  )
}
