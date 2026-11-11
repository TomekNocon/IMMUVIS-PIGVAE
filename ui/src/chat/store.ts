// The connected language model. It lives outside React state so the connection outlives one
// result ("Analyse another image" keeps it).
import { useSyncExternalStore } from 'react'
import type { Connector } from './connectors'

let connected: Connector | null = null
const listeners = new Set<() => void>()

export function setConnected(c: Connector | null) {
  connected = c
  listeners.forEach((l) => l())
}

function subscribe(l: () => void) {
  listeners.add(l)
  return () => {
    listeners.delete(l)
  }
}

export const useConnector = () => useSyncExternalStore(subscribe, () => connected)
