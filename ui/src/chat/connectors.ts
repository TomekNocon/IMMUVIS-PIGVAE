// Ways to connect a language model to the chat. Credentials stay in this browser: the local
// bridge holds the user's own CLI login, and an API key is sent only to the provider.
export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface Connector {
  /** Shown in the panel, e.g. "Claude (local bridge)". */
  label: string
  send(system: string, context: object, messages: ChatMessage[]): Promise<string>
}

export class ChatError extends Error {}

// In development the Vite dev server proxies /bridge to the bridge (vite.config.ts); a built
// site calls the bridge on the user's own computer directly.
export const DEFAULT_BRIDGE_URL = import.meta.env.DEV ? '/bridge' : 'http://127.0.0.1:8765'
export const DEFAULT_OPENAI_MODEL = 'gpt-4o-mini'
const TIMEOUT_MS = 200_000 // the bridge allows a CLI call up to 180 s

async function request(url: string, init: RequestInit, what: string): Promise<Response> {
  try {
    return await fetch(url, { ...init, signal: AbortSignal.timeout(TIMEOUT_MS) })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'TimeoutError') throw new ChatError(`${what} did not answer in time.`)
    throw new ChatError(`Cannot reach ${what}.`)
  }
}

/** The local bridge (scripts/deploy/chat_bridge.py) running Codex or Claude with your own login. */
export async function connectBridge(baseUrl: string): Promise<Connector> {
  const base = baseUrl.replace(/\/$/, '')
  const what = `the local bridge at ${base}`
  const health = await request(`${base}/health`, {}, what)
  if (!health.ok) throw new ChatError(`${what} refused the connection (${health.status}).`)
  const info = (await health.json()) as { backend: string; model: string | null }
  const name = info.backend === 'codex' ? 'ChatGPT via Codex' : info.backend === 'claude' ? 'Claude' : info.backend
  return {
    label: `${name}${info.model ? ` · ${info.model}` : ''} (local bridge)`,
    async send(system, context, messages) {
      const res = await request(
        `${base}/chat`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ system, context, messages }) },
        what,
      )
      const body = await res.json().catch(() => ({}))
      if (!res.ok) throw new ChatError(body.error ?? `The bridge returned ${res.status}.`)
      return body.reply as string
    },
  }
}

/** OpenAI with the user's own API key, called straight from the browser. */
export async function connectOpenAI(apiKey: string, model: string): Promise<Connector> {
  const headers = { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' }
  // Listing models checks the key without spending anything.
  const check = await request('https://api.openai.com/v1/models', { headers }, 'OpenAI')
  if (check.status === 401) throw new ChatError('OpenAI rejected this API key.')
  if (!check.ok) throw new ChatError(`OpenAI returned ${check.status}.`)
  return {
    label: `OpenAI · ${model}`,
    async send(system, context, messages) {
      const res = await request(
        'https://api.openai.com/v1/chat/completions',
        {
          method: 'POST',
          headers,
          body: JSON.stringify({
            model,
            messages: [
              { role: 'system', content: `${system}\n\n<model_output>\n${JSON.stringify(context)}\n</model_output>` },
              ...messages,
            ],
          }),
        },
        'OpenAI',
      )
      const body = await res.json().catch(() => ({}))
      if (!res.ok) {
        if (res.status === 429) throw new ChatError('OpenAI: rate limit or no credit left on this key.')
        throw new ChatError(`OpenAI: ${body.error?.message ?? res.status}`)
      }
      return body.choices?.[0]?.message?.content ?? ''
    },
  }
}

/** Mock mode only: canned answers, so the panel can be tried without any model. */
export function demoConnector(): Connector {
  return {
    label: 'Demo replies (mock mode)',
    async send(_system, context, messages) {
      await new Promise((r) => setTimeout(r, 900))
      const c = context as { prediction: string; crops: { row: number; col: number; attention: number }[] }
      const top = c.crops[0]
      return `(demo reply) You asked: "${messages[messages.length - 1].content}". The model predicts ${c.prediction}; its highest-attention crop is row ${top.row}, col ${top.col} with ${(top.attention * 100).toFixed(1)}% of the attention. Connect a real model to get actual answers.`
    },
  }
}

// ---- remembered settings (never the API key unless the user asks) ----------------------------

export interface ChatSettings {
  kind: 'bridge' | 'openai' | 'demo'
  bridgeUrl: string
  model: string
  remember: boolean
}

const SETTINGS_KEY = 'imc-chat-settings'
const KEY_KEY = 'imc-chat-openai-key'

export function loadSettings(): ChatSettings {
  const fallback: ChatSettings = { kind: 'bridge', bridgeUrl: DEFAULT_BRIDGE_URL, model: DEFAULT_OPENAI_MODEL, remember: false }
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}') as Partial<ChatSettings>
    if (!saved.bridgeUrl) delete saved.bridgeUrl
    return { ...fallback, ...saved }
  } catch {
    return fallback
  }
}

export function saveSettings(s: ChatSettings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s))
  } catch {
    // storage blocked: settings just aren't remembered
  }
}

export function loadKey(): string {
  try {
    return sessionStorage.getItem(KEY_KEY) ?? localStorage.getItem(KEY_KEY) ?? ''
  } catch {
    return ''
  }
}

/** Session storage by default (gone when the tab closes); local storage only if remembered. */
export function saveKey(key: string, remember: boolean): void {
  try {
    sessionStorage.setItem(KEY_KEY, key)
    if (remember) localStorage.setItem(KEY_KEY, key)
    else localStorage.removeItem(KEY_KEY)
  } catch {
    // storage blocked: the key lives only in memory for this page
  }
}

export function forgetKey(): void {
  try {
    sessionStorage.removeItem(KEY_KEY)
    localStorage.removeItem(KEY_KEY)
  } catch {
    // ignore
  }
}
