import { STARTERS } from './prompt'
import './chat.css'

const QUICK: [label: string, starter: number][] = [
  ['Why this prediction?', 0],
  ['Most important crop', 1],
  ['How confident?', 2],
]

/** Quick questions under the prediction; each opens the drawer and asks. */
export function AskBar({ onAsk, onOpen }: { onAsk: (question: string) => void; onOpen: () => void }) {
  return (
    <div>
      <div className="caps head">Ask about this result</div>
      <div className="askLinks">
        {QUICK.map(([label, i]) => (
          <button key={label} className="askLink" onClick={() => onAsk(STARTERS[i])}>
            {label}
          </button>
        ))}
        <button className="askLink own" onClick={onOpen}>
          Ask your own →
        </button>
      </div>
    </div>
  )
}
