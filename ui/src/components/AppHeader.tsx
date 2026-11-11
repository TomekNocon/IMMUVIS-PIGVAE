import { USING_MOCK } from '@/api'

interface Props {
  theme: 'dark' | 'light'
  onToggleTheme: () => void
  /** Shown only while signed in. */
  onSignOut?: () => void
}

export function AppHeader({ theme, onToggleTheme, onSignOut }: Props) {
  return (
    <header className="hdr">
      <div className="brand">
        <span className="serif brandName">IMC subtype viewer</span>
        <span className="hint">lung · adenocarcinoma / squamous</span>
      </div>
      <div className="hdrActions">
        {USING_MOCK && <span className="mockTag">mock data</span>}
        <button className="hbtn" onClick={onToggleTheme}>
          {theme === 'dark' ? 'Light mode' : 'Dark mode'}
        </button>
        {onSignOut && (
          <button className="hbtn" onClick={onSignOut}>
            Sign out
          </button>
        )}
      </div>
    </header>
  )
}
