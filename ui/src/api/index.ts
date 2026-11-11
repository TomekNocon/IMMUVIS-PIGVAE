// The one entry point the UI imports: `import { api } from '@/api'`.
// VITE_USE_MOCK=1 swaps in recorded responses; the mock code is left out of real builds.
import { mockApi } from './mock'
import { realApi } from './client'
import type { Api } from './types'

export const USING_MOCK = import.meta.env.VITE_USE_MOCK === '1'

export const api: Api = USING_MOCK ? mockApi : realApi

export * from './types'
