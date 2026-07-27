# Wiki Write-Back

How to record a finding in `research-ml` so it matches the existing structure.
The wiki is an Obsidian-style vault: append-only `log.md`, a `synthesis.md` that
is rebuilt incrementally, an `index.md` catalog, and `wiki/` pages cross-linked
with `[[wikilinks]]`. **Draft everything, show the user, write only on approval.**
Never auto-write. Convert relative dates to absolute (today is in context).

A full "ingest" of one result usually touches **five** things: a `source-*` page
(raw facts), an `exp-*` page (interpretation), an `experiment-log.md` row, a
`synthesis.md` update, and `index.md` links — plus the `log.md` entry recording
the op. For a small/quick analysis, a `log.md` entry alone may be enough; scale to
the finding and ask the user how deep to record it.

Read the two most recent real examples before drafting, to match voice and depth:
`wiki/sources/source-vae16-overfit.md` and
`wiki/experiments/exp-2026-06-24-vae16-overfit-mechanism.md`.

## Slugs and the link graph

- Slug: short, kebab-case, descriptive of the *finding* (`vae16-fb-sweep`), not the
  run id. exp page → `wiki/experiments/exp-<YYYY-MM-DD>-<slug>.md`; source page →
  `wiki/sources/source-<slug>.md`.
- Every page links its neighbours: exp `source:` → its source page; source pages
  link sibling sources; synthesis claims link the exp/source that back them.
- After creating a page, add it to `index.md` *and* link it from wherever it's
  relevant (synthesis, experiment-log). An orphan page is a bug.

## 1. `log.md` — append-only op entry

Append at the end. Format (kept grep-able by `grep "^## \[" log.md`):

```markdown
## [YYYY-MM-DD] <op> | <title>
- pages touched: [[page-a]], [[page-b]], [[index]], [[synthesis]]
- note: 2–5 sentences — what was investigated (name the wandb run id + entity),
  the finding, and the decision/next lever. Terse, factual, past tense.
```

`<op>` ∈ `init | plan | query | ingest | edit`. A run analysis recorded into the
wiki is usually **ingest**; a pure diagnosis with no page changes is **query**.

## 2. `wiki/sources/source-<slug>.md` — the raw facts

The auditable record: config + curves + latent end-state + diagnose numbers, with
minimal interpretation. Frontmatter:

```markdown
---
type: source
title: Run <name> (<run-id>) — <one-line>
source_type: experiment
created: YYYY-MM-DD
updated: YYYY-MM-DD
link: https://wandb.ai/tomeknocon/PIGVAE/runs/<run-id>
tags: [vae, kld, free-bits, ..., wandb]
---

## Summary
2–4 sentences: what the run is, the headline result, the comparison to its sibling.

## Key points
- **`<run-id>` = <full config name>** (z, KL scale, free_bits, α schedule, epochs,
  batch). Project `PIGVAE`, entity `tomeknocon`.
  - val_mse: trajectory + best @ epoch + final + degradation factor.
  - train mse: flat value; train↔val gap.
  - latent end-state: kld/dim, std_mean (σ), mu_std, active_dims/total.
  - **Diagnose on `<ckpt>`:** residual max_abs, QK gains, image-space floor vs
    model error, R² — only if the offline inspector was run.
```

Numbers go here so the exp page can stay about *meaning*.

## 3. `wiki/experiments/exp-<YYYY-MM-DD>-<slug>.md` — the interpretation

The argument: what happened and *why*, with tables that carry the evidence.
Frontmatter + canonical sections:

```markdown
---
type: experiment
title: <plain-language finding>
created: YYYY-MM-DD
updated: YYYY-MM-DD
source: "[[source-<slug>]]"
tags: [experiment, vae, ...]
---

# <title>

1–2 sentence abstract: the finding and why it matters, linking [[source-<slug>]].

## Setup
- **Model** [[pi-gvae]] ... data, resolution, KL recipe, eval convention.
- **`<run-id>`** (name): the levers. + sibling/baseline runs for context.

## Result
The smoking-gun trace — a small table of epoch × {α, kld/dim, σ, μ_std, train,
val}. Bold the key rows and say in one line how to read them.
(For an ablation, a second table comparing the runs side by side.)

## Interpretation — <mechanism>
Numbered mechanisms. Each: the causal story tying the metric to the behaviour
(cloud overlap, loose bottleneck, etc.), not just the symptom.

## Decision / next lever
The single lever and why it beats the alternatives; tie to a [[synthesis]] question.
```

## 4. `wiki/experiments/experiment-log.md` — one table row

Prepend a row at the **top** of the table body (newest first), matching columns:

```
| YYYY-MM-DD | `<run-id>` <slug> | **<model/res>** z<..> (<dims>), KL `scale ..` + `fb ..`, α schedule; per-view | <protocol> | <metric> | **<headline result + key latent numbers>** | [[exp-<...>]] |
```

Keep the cell prose dense and bold the headline number — match the existing rows.

## 5. `synthesis.md` — fold the finding into the thesis

`synthesis.md` is the heart and is rebuilt incrementally. Update the relevant part:
- **Working hypotheses** — if the finding advances the substrate→invariance→transfer
  story, extend the current paragraph (don't append a disconnected one).
- **Evidence for / against / open tensions** — add or sharpen a bullet, each
  linking its `[[source]]`/`[[exp]]`.
- **Open questions** — answer one (and/or add the next), phrased against the
  *deciding metric* (transfer, not recon).
- Bump `updated:` and add the new source to the `sources:` frontmatter list.

Every claim must link the source that backs it. Don't overwrite prior nuance —
integrate. This is the most editorial step; show the diff and let the user steer.

## 6. `index.md` — register the new pages

Add one line each under the right heading, `[[link]] — summary` form:
- under **## Experiments**: `- [[exp-<...>]] — <one-line finding>.`
- under **## Sources**: `- [[source-<slug>]] — wandb run \`<id>\`; <one-line>.`
Bump `updated:`.

## Order of operations when writing (after approval)

source page → exp page → experiment-log row → synthesis update → index links →
log.md entry (records the whole op last). Then offer to `git commit` the wiki
(the user controls commits — ask first).
