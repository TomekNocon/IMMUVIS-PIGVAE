"""Pivot per-source ABMIL results into one comparison table (stage 4 of transfer).

Reads ``results/<source_tag>/results.csv`` for each per-crop feature (raw | zglobal |
node_mean | node_flatten) and writes one wide table indexed by (dataset, feature) with a column
per source, so encoder features can be read off against the raw-768 baseline row by row.

Usage:
    uv run python scripts/downstream/collect_downstream_table.py run_tag=<RUN_TAG> metric=auc
"""

import glob
import os

import hydra
import pandas as pd
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

_METRICS = ("auc", "accuracy", "macro_f1", "precision", "recall")

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


@hydra.main(version_base="1.3", config_path="../../configs", config_name="downstream/collect")
def main(cfg: DictConfig) -> None:
    if cfg.metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {cfg.metric!r}")
    wide = collect(cfg.results_root, cfg.metric)
    out = os.path.join(cfg.results_root, f"comparison_{cfg.metric}.csv")
    wide.to_csv(out, index=False)
    print(f"=== {cfg.run_tag}  ({cfg.metric}, per-crop feature source) ===")
    print(wide.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
