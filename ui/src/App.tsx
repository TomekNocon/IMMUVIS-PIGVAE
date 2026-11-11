// IMC Subtype Viewer: app shell, screen flow, theme and error mapping.
// Design: Claude Design handoff "IMC Subtype Viewer" (product brief: docs/design/2026-11-11-imc-web-ui-brief.md).
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api, ApiError, UnauthorizedError, type PredictResponse } from '@/api'
import { AppHeader } from '@/components/AppHeader'
import { Login } from '@/screens/Login'
import { Predicting } from '@/screens/Predicting'
import { Result } from '@/screens/Result'
import { Upload, type UploadError } from '@/screens/Upload'

type Screen = 'login' | 'upload' | 'predict' | 'result'
type Theme = 'dark' | 'light'

const THEME_KEY = 'imc-viewer-theme'
const EXPIRED = 'Your session expired. Sign in again to continue.'

function loadTheme(): Theme {
  try {
    return localStorage.getItem(THEME_KEY) === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

function uploadError(err: unknown): UploadError {
  if (err instanceof ApiError) {
    if (err.status === 422) return { title: "This image can't be analysed", detail: err.message }
    if (err.status === 413) return { title: 'File is larger than 300 MB', detail: err.message }
    if (err.status === 0) {
      return { title: 'The server is starting, this can take a minute', detail: err.message, retry: true }
    }
    return { title: 'Something went wrong', detail: `${err.status} · ${err.message}`, retry: true }
  }
  return { title: 'Something went wrong', detail: String(err), retry: true }
}

export default function App() {
  const [screen, setScreen] = useState<Screen>(api.isLoggedIn() ? 'upload' : 'login')
  const [theme, setTheme] = useState<Theme>(loadTheme)
  const [note, setNote] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [error, setError] = useState<UploadError | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const queryClient = useQueryClient()

  useEffect(() => {
    document.body.dataset.theme = theme
    try {
      localStorage.setItem(THEME_KEY, theme)
    } catch {
      // storage blocked: the theme still applies for this visit
    }
  }, [theme])

  const signedIn = screen !== 'login'
  // Model card; requesting it also wakes the model server while the user picks a file.
  const info = useQuery({
    queryKey: ['info'],
    queryFn: api.getInfo,
    enabled: signedIn,
    staleTime: Infinity,
    retry: (n, err) => !(err instanceof UnauthorizedError) && n < 4,
    retryDelay: 5000,
  })

  const expire = () => {
    api.logout()
    setNote(EXPIRED)
    setScreen('login')
  }
  // A 401 while loading the model card means the session ran out: show the sign-in screen.
  const sessionLost = info.error instanceof UnauthorizedError
  const shown: Screen = sessionLost ? 'login' : screen

  const predict = useMutation({
    mutationFn: (f: File) => api.predict(f),
    onMutate: () => {
      setError(null)
      setScreen('predict')
    },
    onSuccess: (r) => {
      setResult(r)
      setScreen('result')
    },
    onError: (err) => {
      if (err instanceof UnauthorizedError) return expire()
      setError(uploadError(err))
      setScreen('upload')
    },
  })

  const analyse = (f: File) => {
    setFile(f)
    predict.mutate(f)
  }
  const signOut = () => {
    api.logout()
    setNote(null)
    setResult(null)
    setFile(null)
    setError(null)
    setScreen('login')
  }

  return (
    <div className="app">
      <AppHeader
        theme={theme}
        onToggleTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        onSignOut={shown !== 'login' ? signOut : undefined}
      />

      {shown === 'login' && (
        <Login
          note={sessionLost ? EXPIRED : note}
          onDone={() => {
            setNote(null)
            setScreen('upload')
            queryClient.resetQueries({ queryKey: ['info'] })
          }}
        />
      )}
      {shown === 'upload' && (
        <Upload
          info={info.data}
          serverReady={info.isSuccess}
          file={file}
          error={error}
          onFile={(f, err) => {
            setFile(f)
            setError(err)
          }}
          onAnalyse={analyse}
        />
      )}
      {shown === 'predict' && (
        <Predicting fileName={file?.name ?? ''} expectedSeconds={info.data?.device === 'cuda' ? 10 : 30} />
      )}
      {shown === 'result' && result && (
        <Result
          result={result}
          info={info.data}
          onAgain={() => {
            setFile(null)
            setResult(null)
            setScreen('upload')
          }}
        />
      )}
    </div>
  )
}
