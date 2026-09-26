# scripts/collect_downstream_table.py
"""Collect per-source ABMIL results into one comparison table (Stage 4).

The feature-source sweep writes one `results/<source_tag>/results.csv` per
per-crop feature (raw | zglobal | node_mean | node_flatten), each with columns
`dataset, feature, accuracy, macro_f1, auc`. This pivots them into a single wide
table indexed by (dataset, feature) with one column per source, so the deciding
question -- does the trained encoder's z_global beat the raw-768 baseline? -- is
a single glance across a row.

Example:
  uv run python scripts/collect_downstream_table.py --run_tag vae16_s0p001_2026-06-28
"""
import argparse
import glob
import os

import pandas as pd

# Stable column order for the sources we sweep (any missing source is skipped).
_SOURCE_ORDER = ["raw", "zglobal", "node_mean", "node_flatten"]


def collect(results_root: str, metric: str = "auc") -> pd.DataFrame:
    """Pivot every `<source>/results.csv` under `results_root` into a wide table."""
    frames = []
    for csv in sorted(glob.glob(os.path.join(results_root, "*", "results.csv"))):
        source = os.path.basename(os.path.dirname(csv))
        df = pd.read_csv(csv)
        if metric not in df.columns:
            # A source whose CV predates this metric (e.g. precision/recall added later, or only
            # some sources re-run) simply has no column -- skip it rather than abort the whole
            # table, so a partial metric still collects the sources that do have it.
            print(f"[skip] {source}: no '{metric}' column (columns: {list(df.columns)})")
            continue
        df = df[["dataset", "feature", metric]].rename(columns={metric: source})
        frames.append(df.set_index(["dataset", "feature"]))
    if not frames:
        raise FileNotFoundError(f"no <source>/results.csv found under {results_root}")
    wide = pd.concat(frames, axis=1)
    ordered = [c for c in _SOURCE_ORDER if c in wide.columns]
    ordered += [c for c in wide.columns if c not in ordered]  # keep any extra sources
    return wide[ordered].reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect per-source ABMIL results into one table.")
    ap.add_argument("--run_tag", required=True)
    ap.add_argument("--downstream_root",
                    default="/raid_encrypted/immucan/embeddings/tnocon/downstream")
    ap.add_argument("--metric", default="auc",
                    choices=["auc", "accuracy", "macro_f1", "precision", "recall"])
    args = ap.parse_args()

    results_root = os.path.join(args.downstream_root, args.run_tag, "results")
    wide = collect(results_root, args.metric)
    out = os.path.join(results_root, f"comparison_{args.metric}.csv")
    wide.to_csv(out, index=False)
    print(f"=== {args.run_tag}  ({args.metric}, per-crop feature source) ===")
    print(wide.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
