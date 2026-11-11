// The real API client: talks to the Modal gateway at VITE_API_URL.
import { clearSession, getToken, setSession } from './session'
import {
  ApiError,
  MAX_UPLOAD_BYTES,
  UnauthorizedError,
  type Api,
  type InfoResponse,
  type LoginResponse,
  type PredictOptions,
  type PredictResponse,
} from './types'

// In development the Vite dev server proxies /api to VITE_API_URL (see vite.config.ts);
// built sites call the gateway directly.
const BASE = import.meta.env.DEV ? '' : (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
// Give up on a request after this long (a cold start plus a CPU prediction is ~50 s).
const TIMEOUT_MS = 150_000

// FastAPI errors are {detail: string} or, for request validation, {detail: [{msg, loc}, ...]}.
async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) {
      return body.detail.map((d: { msg?: string }) => d.msg ?? String(d)).join('; ')
    }
  } catch {
    // not JSON
  }
  return `${res.status} ${res.statusText}`
}

async function call<T>(path: string, init: RequestInit = {}, auth = true): Promise<T> {
  const headers = new Headers(init.headers)
  if (auth) {
    const token = getToken()
    if (!token) throw new UnauthorizedError()
    headers.set('Authorization', `Bearer ${token}`)
  }
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers, signal: AbortSignal.timeout(TIMEOUT_MS) })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'TimeoutError') {
      throw new ApiError(0, 'No answer from the server within 2.5 minutes.')
    }
    throw new ApiError(0, 'Cannot reach the server. It may be starting up; try again in a minute.')
  }
  if (res.status === 401 && auth) {
    clearSession()
    throw new UnauthorizedError()
  }
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res))
  return (await res.json()) as T
}

export const realApi: Api = {
  async login(password) {
    const body = await call<LoginResponse>(
      '/api/login',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      },
      false,
    )
    setSession(body.token, body.expires_at)
    return body
  },

  logout: clearSession,

  isLoggedIn: () => getToken() !== null,

  getInfo: () => call<InfoResponse>('/api/info'),

  async predict(file: File, options: PredictOptions = {}) {
    if (file.size > MAX_UPLOAD_BYTES) throw new ApiError(413, 'File is larger than 300 MB.')
    const params = new URLSearchParams()
    if (options.igSteps !== undefined) params.set('ig_steps', String(options.igSteps))
    if (options.previewMarkers) params.set('preview_markers', options.previewMarkers.join(','))
    const form = new FormData()
    form.append('file', file)
    const query = params.size ? `?${params}` : ''
    return call<PredictResponse>(`/api/predict${query}`, { method: 'POST', body: form })
  },
}
