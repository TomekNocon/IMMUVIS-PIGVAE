"""Join clinical labels onto the encode-stage crop metadata (stage 2 of transfer).

The encode stage writes per-crop metadata (img_path, coords, embeddings_file, embedding_idx);
this attaches the clinical labels from the melted table by tiff basename, giving the columns
``run_abmil.py`` consumes: img_path, coords*, embeddings_file, embedding_idx, dataset,
image_shape, feature, feature_value. Mirrors ``merge_metadata_with_melted`` of the gated-ABMIL
reference pipeline.

Usage:
    uv run python scripts/downstream/build_abmil_meta.py run_tag=<RUN_TAG> source_tag=zglobal
"""

import os

import hydra
import pandas as pd
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _img_filename(path: str) -> str:
    """main.py::merge_metadata_with_melted's inner normalizer: path -> tiff basename."""
    name = str(path).split("/")[-1]
    if ".patch" in name:
        return name.split(".patch")[0] + ".tiff"
    return name


def merge_split(encode_meta_csv: str, melted: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    """Attach clinical labels to one split's encode metadata; write + return the join."""
    emb = pd.read_csv(encode_meta_csv)
    img_col = "img_path" if "img_path" in emb.columns else "image_paths"
    emb = emb.copy()
    emb["img_path"] = emb[img_col].apply(_img_filename)
    merged = pd.merge(emb, melted, left_on="img_path", right_on="image_path")
    merged = merged.drop(columns=["panel", "image_path"], errors="ignore")
    if len(merged) == 0:
        raise ValueError(
            f"zero rows after join for {encode_meta_csv}: no encode img_path basename "
            f"matched the melted table's image_path (e.g. '7748.tiff')."
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


@hydra.main(version_base="1.3", config_path="../../configs", config_name="downstream/build_meta")
def main(cfg: DictConfig) -> None:
    melted = pd.read_csv(cfg.melted_csv)
    melted = melted[melted["dataset"] == cfg.dataset]
    if len(melted) == 0:
        raise ValueError(f"no dataset=={cfg.dataset!r} rows in {cfg.melted_csv}")

    for split in cfg.splits:
        src = os.path.join(cfg.encode_dir, f"{cfg.dataset}_{split}_metadata.csv")
        out = os.path.join(cfg.out_dir, f"{cfg.dataset}_{split}_metadata.csv")
        merged = merge_split(src, melted, out)
        print(
            f"{split}: {merged['img_path'].nunique()} imgs x "
            f"{merged['feature'].nunique()} features = {len(merged)} rows -> {out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
