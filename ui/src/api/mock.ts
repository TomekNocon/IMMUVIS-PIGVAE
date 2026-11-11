// Mock API for design work (VITE_USE_MOCK=1): answers with responses recorded from the real
// server, after realistic delays. No network, no token, no cost.
//
// Steering the mock:
//   login password "wrong"  -> 401 wrong password;  "locked" -> 429 too many attempts
//   upload a file whose name contains "squam"  -> Squamous cell carcinoma result
//   upload a file whose name contains "bad", or not .tif/.tiff/.npy -> 422 error
//   any other image file -> Adenocarcinoma result
import { clearSession, getToken, setSession } from './session'
import {
  ApiError,
  MAX_UPLOAD_BYTES,
  UnauthorizedError,
  type Api,
  type InfoResponse,
  type PredictResponse,
} from './types'

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

function requireSession(): void {
  if (!getToken()) throw new UnauthorizedError()
}

export const mockApi: Api = {
  async login(password) {
    await sleep(400)
    if (password === 'locked') {
      throw new ApiError(429, 'too many failed attempts, try again in a few minutes')
    }
    if (!password || password === 'wrong') throw new ApiError(401, 'wrong password')
    const expiresAt = Math.floor(Date.now() / 1000) + 8 * 3600
    setSession('mock-token', expiresAt)
    return { token: 'mock-token', expires_at: expiresAt }
  },

  logout: clearSession,

  isLoggedIn: () => getToken() !== null,

  async getInfo() {
    await sleep(600)
    requireSession()
    return (await import('./fixtures/info.json')).default as InfoResponse
  },

  async predict(file) {
    requireSession()
    if (file.size > MAX_UPLOAD_BYTES) throw new ApiError(413, 'File is larger than 300 MB.')
    await sleep(3000)
    const name = file.name.toLowerCase()
    if (name.includes('bad') || !/\.(tiff?|npy)$/.test(name)) {
      const err = (await import('./fixtures/error_422.json')).default
      throw new ApiError(422, err.detail)
    }
    const fixture = name.includes('squam')
      ? await import('./fixtures/predict_squamous.json')
      : await import('./fixtures/predict_adenocarcinoma.json')
    return fixture.default as PredictResponse
  },
}
