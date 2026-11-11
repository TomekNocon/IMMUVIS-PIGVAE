// Named types for the browser API. The source of truth is the Python backend
// (src/deploy/schemas.py -> src/deploy/openapi.json -> schema.ts via `pnpm gen:api`).
// Do not edit schema.ts by hand.
import type { components } from './schema'

type S = components['schemas']

export type LoginResponse = S['LoginResponse']
export type InfoResponse = S['InfoResponse']
export type PredictResponse = S['PredictResponse']
export type Crop = S['Crop']
export type Preview = S['Preview']
export type ImageInfo = S['ImageInfo']
export type ModelInfo = S['ModelInfo']

export interface PredictOptions {
  /** Integrated-Gradients steps, 0..500; 0 skips IG (crop.ig_importance is then null). Default 50. */
  igSteps?: number
  /** Three marker names shown as red, green, blue in the preview. Default panCK, CD45, DNA1. */
  previewMarkers?: [string, string, string]
}

/** Everything the UI may call. Real and mock implementations share it. */
export interface Api {
  /** Log in; keeps the session token for later calls. Throws ApiError (401 wrong password, 429 locked). */
  login(password: string): Promise<LoginResponse>
  logout(): void
  isLoggedIn(): boolean
  /** Model card and expected input. Also wakes the model server, so call it early. */
  getInfo(): Promise<InfoResponse>
  /** Upload an image and get the prediction. Takes ~30 s on the current CPU server. */
  predict(file: File, options?: PredictOptions): Promise<PredictResponse>
}

/** Any failed call. `status` is the HTTP status, or 0 when the server could not be reached. */
export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** The session is missing or expired: show the login screen. */
export class UnauthorizedError extends ApiError {
  constructor(message = 'Your session has expired, please log in again.') {
    super(401, message)
    this.name = 'UnauthorizedError'
  }
}

/** Largest upload the server accepts (300 MB). */
export const MAX_UPLOAD_BYTES = 300 * 1024 ** 2
