# IMC subtype viewer (UI)

Browser front end for the IMC inference API on Modal. Design: `../docs/design/2026-11-11-imc-web-ui.md`;
product brief (what the numbers mean, error states): `../docs/design/2026-11-11-imc-web-ui-brief.md`;
visual design: the Claude Design project "IMC Subtype Viewer".

## Run (on bury)

```bash
cd ui
pnpm install       # first time only
pnpm dev:mock      # recorded responses: no server, any password, instant
pnpm dev           # the real API (VITE_API_URL in .env.local); sign in with UI_PASSWORD from ../.env
pnpm build         # type-check and build into dist/
pnpm lint
```

Open the forwarded port 5173 on your laptop (VS Code -> Ports tab) as `http://localhost:5173`
(the API only accepts that exact origin in development). Stop with Ctrl+C.

Mock mode: password `wrong` -> 401, `locked` -> 429, anything else signs in. The Upload screen
shows two example links; a file whose name contains `bad` gives the 422 error state.

## Layout

| Path | What |
|------|------|
| `src/api/` | **The contract**: client, mock, generated types, fixtures. Changed only together with the backend. The fixtures are real API responses for images 8080 and 7810 (held out from fold 3) with **synthetic preview images**: the real previews are patient-derived and stay off git. |
| `src/App.tsx` | Screen flow (login -> upload -> predicting -> result), theme, error mapping |
| `src/screens/` | The four screens |
| `src/result/` | Result screen: `Viewer` (image, heatmap, zoom/pan, compare), `Panel` (prediction, charts, top crops), `model.ts` (derived data, formatting, class colours) |
| `src/app.css` | Design tokens (light/dark) and shared styles |

The UI talks to the server only through `import { api, ... } from '@/api'`.

## When the backend API changes

```bash
uv run python scripts/deploy/export_openapi.py   # repo root: Python models -> src/deploy/openapi.json
cd ui && pnpm gen:api                            # -> src/api/schema.ts (TypeScript types)
```

`tests/test_deploy_contract.py` fails if `openapi.json` or the fixtures drift from the backend.

## Deploy

Any static host works (Vercel, Cloudflare Pages): root directory `ui`, build command `pnpm build`,
output `dist`, environment variable `VITE_API_URL`. Then add the site's address to `UI_ORIGINS`
in the Modal Secret `pigvae-imc-ui` so the API accepts calls from it.

## Notes

- `pnpm-workspace.yaml` pins `rolldown` (Vite's bundler) to 1.2.19: 1.2.18 crashes with
  "Illegal instruction" on bury's older CPU.
- TypeScript is 5.9, not the template's 7.0, because `openapi-typescript` needs 5.
