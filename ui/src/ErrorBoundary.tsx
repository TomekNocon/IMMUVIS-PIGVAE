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
      <div className="app crash">
        <div className="crashBody">
          <h1 className="title">Something went wrong</h1>
          <div className="mono crashMsg">{error.message}</div>
          <div className="mut crashText">
            Reloading the page usually helps; you stay signed in. If it happens again, the browser console
            (F12) has the details.
          </div>
          <button className="primary alignStart" onClick={() => location.reload()}>
            Reload
          </button>
        </div>
      </div>
    )
  }
}
