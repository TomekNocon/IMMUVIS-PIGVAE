# IMC subtype viewer (UI)

Browser front end for the IMC inference API on Modal (`src/deploy/`): sign in, upload a raw cords
IMC image, see the DX.name prediction with per-crop attention and Integrated Gradients. Visual
design: the Claude Design project "IMC Subtype Viewer".

## Run

From the repository root (or the `pnpm` command inside `ui/`):

| make | pnpm | What |
|------|------|------|
| `make ui-install` | `pnpm install` | install dependencies (`make ui-install` also creates `.env.local` from `.env.example`) |
| `make ui-dev` | `pnpm dev` | the real API; sign in with `UI_PASSWORD` from the repo-root `.env` |
| `make ui-mock` | `pnpm dev:mock` | recorded responses: no server, any password, instant |
| `make ui-build` | `pnpm build && pnpm lint` | type-check, lint, build into `dist/` |

The dev server listens on `localhost:5173`; on the cluster, forward that port (VS Code -> Ports tab)
and open `http://localhost:5173` on your laptop. Stop with Ctrl+C. During development Vite forwards
`/api` to the gateway in `VITE_API_URL` (see `vite.config.ts`), so any local port works.

Mock mode: password `wrong` -> 401, `locked` -> 429, anything else signs in. The Upload screen
shows two example links; a file whose name contains `bad` gives the 422 error state.

## Chat about a result (optional)

On the result screen, the "Ask about this result" bar under the prediction offers quick questions
and "Ask your own"; both open a 400 px chat drawer (Claude Design handoff). On wide screens the page
moves aside instead of being covered; an answer about a crop pins it in the viewer, and crop
mentions ("row 0, col 0") highlight it on hover. A language model gets the prediction's
numbers as JSON (probabilities, logit, per-crop attention and IG, model card; **never the image**)
plus the rules in `src/chat/prompt.ts` (explain this output only, no diagnosis, say what it cannot
know, be concise). Nothing goes through the Modal backend; each user connects their own model:

- **My subscription (local):** run the bridge on the computer your browser runs on, with the
  Codex CLI (`npm i -g @openai/codex && codex login`, ChatGPT plan) or Claude Code (`claude`,
  Claude plan) installed and signed in: `make chat-bridge` (or
  `python3 scripts/deploy/chat_bridge.py`, standard library only). It answers on
  `127.0.0.1:8765`, only to the UI's origins, and runs the CLI without tools in an empty temporary
  folder. With `make ui-dev` the panel's bridge address is `/bridge`: the dev server forwards it,
  so on the cluster the bridge can run next to the dev server and only port 5173 is forwarded.
  A deployed site calls `http://127.0.0.1:8765` on the user's computer; start the bridge there
  with `--allow-origin https://<site>`.
- **OpenAI API key:** paste a key from platform.openai.com (billed per use; a ChatGPT subscription
  is not an API key). It stays in the browser (session storage, or local storage if remembered) and
  goes only to OpenAI.
- **Demo replies:** mock mode only, canned answers.

## Layout

| Path | What |
|------|------|
| `src/api/` | **The contract**: client, mock, generated types, fixtures. Changed only together with the backend. The fixtures are real API responses for images 8080 and 7810 (held out from fold 3) with **synthetic preview images**: the real previews are patient-derived and stay off git. |
| `src/App.tsx` | Screen flow (login -> upload -> predicting -> result), theme, error mapping |
| `src/screens/` | The four screens; each splits into small named components |
| `src/result/` | Result screen: `Viewer` (image, heatmap, pin/compare) with `useZoomPan` (zoom and drag), `Panel` (prediction, charts, top crops), `model.ts` (derived data, formatting, class colours) |
| `src/chat/` | `AskBar`, `ChatDrawer` (`Connect` or `Conversation`), `store.ts` (the connected model), connectors (local bridge, OpenAI key, demo), system prompt and context, Markdown with crop links |
| `src/components/` | Pieces used across screens: `AppHeader`, `Field`, `cx` (class names) |
| `src/app.css` | Design tokens (light/dark) and shared styles (text, buttons, inputs) |

**Styling:** plain CSS. Each component's styles sit next to it (`Upload.tsx` + `Upload.css`);
inline `style` is used only for values computed from the data (crop positions, bar widths, class
colours).

The UI talks to the server only through `import { api, ... } from '@/api'`.

## When the backend API changes

The response shapes are defined once, as Python models in `src/deploy/schemas.py`. After changing
them, run `make modal-api`, which does two steps:

1. `scripts/deploy/export_openapi.py` writes the API description `src/deploy/openapi.json`;
2. `pnpm gen:api` (a script in `package.json`, using `openapi-typescript`) turns that file into
   the TypeScript types `src/api/schema.ts`, which the UI code imports through `src/api/types.ts`.

`tests/test_deploy_contract.py` fails if `openapi.json` or the fixtures drift from the backend.

## Deploy

Any static host works (Vercel, Cloudflare Pages): root directory `ui`, build command `pnpm build`,
output `dist`, environment variable `VITE_API_URL` (the gateway address from `.env.example`). The
built site calls the gateway directly, so allow its address: add it to `UI_ORIGINS` in the
repo-root `.env` (comma-separated), then `make modal-secret FORCE=1 && make modal-deploy`.

## Notes

- `pnpm-workspace.yaml` pins `rolldown` (Vite's bundler) to 1.2.19: 1.2.18 crashes with
  "Illegal instruction" on bury's older CPU.
- TypeScript is 5.9, not the template's 7.0, because `openapi-typescript` needs 5.
