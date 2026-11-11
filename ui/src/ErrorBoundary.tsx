// Shows a crash as a readable message instead of a blank page.
import { Component, type ErrorInfo, type ReactNode } from 'react'

export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('UI crashed:', error, info.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div className="app" style={{ display: 'grid', placeItems: 'center', padding: 32 }}>
        <div style={{ maxWidth: 560, display: 'flex', flexDirection: 'column', gap: 14 }}>
          <h1 className="serif" style={{ fontWeight: 400, fontSize: 32, margin: 0 }}>
            Something went wrong
          </h1>
          <div className="mono" style={{ fontSize: 13, color: 'var(--err)', overflowWrap: 'anywhere' }}>
            {error.message}
          </div>
          <div className="mut" style={{ fontSize: 14 }}>
            Reloading the page usually helps; you stay signed in. If it happens again, the browser console
            (F12) has the details.
          </div>
          <button className="primary" style={{ alignSelf: 'flex-start', padding: '0 18px' }} onClick={() => location.reload()}>
            Reload
          </button>
        </div>
      </div>
    )
  }
}
