# scripts/build_abmil_meta.py
"""Build label-joined ABMIL meta tables (Stage 2 of the PIGVAE->ABMIL pipeline).

Replicates the merge in `/home/tnocon/mil/logistic_regression/main.py`
(`merge_metadata_with_melted`): the PIGVAE encode stage already writes per-crop
metadata [img_path, coords0, coords1, embeddings_file, embedding_idx] with
`embeddings_file`/`embedding_idx` pointing into our z_global .npy, so -- unlike
main.py -- we do NOT re-pair embeddings (that's `concat_metadata`'s job upstream);
we only attach clinical labels.

The join, verbatim from main.py: normalize the encode `img_path` to its tiff
basename via `_img_filename`, then `merge(left_on="img_path",
right_on="image_path")` against the melted clinical table filtered to `dataset`.
Output columns (img_path, coords*, embeddings_file, embedding_idx, dataset,
image_shape, feature, feature_value) are exactly what `scripts/run_abmil.py`
consumes (it selects rows per `feature`).

Example:
  uv run python scripts/build_abmil_meta.py \
    --encode_dir /raid_encrypted/immucan/embeddings/tnocon/downstream/<RUN>/mil_embeddings \
    --melted /raid_encrypted/immucan/embeddings/tnocon/data/clinical/melted_table_images.csv \
    --dataset cords \
    --out_dir /raid_encrypted/immucan/embeddings/tnocon/downstream/<RUN>/meta_tables \
    --splits train test
"""
import argparse
import os

import pandas as pd


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Join clinical labels onto PIGVAE encode metadata.")
    ap.add_argument("--encode_dir", required=True,
                    help="dir with <dataset>_<split>_metadata.csv from encode_mil_embeddings.py")
    ap.add_argument("--melted", required=True,
                    help="melted_table_images.csv (long: dataset,image_path,image_shape,feature,feature_value)")
    ap.add_argument("--dataset", default="cords")
    ap.add_argument("--out_dir", required=True, help="destination dir for the joined meta tables")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    melted = pd.read_csv(args.melted)
    melted = melted[melted["dataset"] == args.dataset]
    if len(melted) == 0:
        raise ValueError(f"no dataset=={args.dataset!r} rows in {args.melted}")

    for split in args.splits:
        src = os.path.join(args.encode_dir, f"{args.dataset}_{split}_metadata.csv")
        out = os.path.join(args.out_dir, f"{args.dataset}_{split}_metadata.csv")
        merged = merge_split(src, melted, out)
        print(
            f"{split}: {merged['img_path'].nunique()} imgs x "
            f"{merged['feature'].nunique()} features = {len(merged)} rows -> {out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
