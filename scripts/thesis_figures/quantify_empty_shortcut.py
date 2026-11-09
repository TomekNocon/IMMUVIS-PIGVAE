"""Quantify the ABMIL empty-crop shortcut without retraining (open item 15).

For the reference run's z_global downstream (vae16_s0p01_film_2026-07-31, source zglobal):
  1. tissue density of every crop from the raw TIFF: mean arcsinh(x/5) over channels and pixels;
     a crop is EMPTY if density <= max(0.1, 0.4 * median density of its image's crops)
     (the rule of the former raw-image visualiser); a stricter absolute rule (<= 0.1) is
     reported as a sensitivity check.
  2. rebuild each label's bags and the 10 stratified folds exactly as run_abmil/run_cv do
     (same filtering, bag order, StratifiedKFold(random_state=42), train-fold z-score), load each
     fold's saved head, and on the held-out bags record
       - the attention mass on empty crops (mean over the 8 heads, summed over empty crops),
       - the out-of-fold AUC with all crops (must reproduce the reported AUC) and with empty crops
         masked out of the attention softmax (same frozen head, no retraining).
"""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, tifffile, torch
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from src.downstream.abmil.cv import _auc, zscore_stats_from_bags, apply_zscore_to_bags
from src.downstream.abmil.data import build_image_bags
from src.downstream.abmil.lit import AbmilLitModule
from run_abmil import drop_nan_labels, balance_meta_df, _load_pooled_meta

RUN = "/raid_encrypted/immucan/embeddings/tnocon/downstream/vae16_s0p01_film_2026-07-31"
META, RES = f"{RUN}/meta_tables/zglobal", f"{RUN}/results/zglobal/cords"
TIFF = "/raid_encrypted/immucan/IMC/immu-vis-other-ds/full-20122025"
OUT = "figures/_cache/empty_shortcut"
FEATURES = ["Grade", "Relapse", "DX.name", "Stage"]
P = 128


def image_density(args):
    img, coords = args
    a = np.arcsinh(tifffile.imread(os.path.join(TIFF, img)).astype(np.float32) / 5.0)
    out = []
    for r, c in coords:
        r, c = int(r), int(c)
        out.append(float(a[:, r:r + P, c:c + P].mean()) if r + P <= a.shape[1] and c + P <= a.shape[2] else np.nan)
    return img, out, a.shape


def attention(model, x, mask):
    """GatedABMIL forward that also returns the attention [B, H, S]."""
    v, u = torch.tanh(model.V(x)), torch.sigmoid(model.U(x))
    a = model.W(v * u).masked_fill(mask.unsqueeze(2), -1e9)
    a = torch.softmax(a, dim=1).transpose(1, 2)
    pooled = torch.bmm(a, x).reshape(x.size(0), model.num_heads * x.size(2))
    return model.classifier(pooled), a


def pad(bags, extra_mask=None):
    S = max(b.shape[0] for b in bags); D = bags[0].shape[1]
    x = torch.zeros(len(bags), S, D); m = torch.ones(len(bags), S, dtype=torch.bool)
    for i, b in enumerate(bags):
        x[i, :len(b)] = torch.from_numpy(np.asarray(b, "float32")); m[i, :len(b)] = False
        if extra_mask is not None: m[i, :len(b)] |= torch.from_numpy(extra_mask[i])
    return x, m


def main():
    os.makedirs(OUT, exist_ok=True)
    pooled = _load_pooled_meta(META, "cords")
    crops = pooled.drop_duplicates(["img_path", "coords0", "coords1"])[["img_path", "coords0", "coords1"]]
    dens_file = f"{OUT}/crop_density.csv"
    if not os.path.exists(dens_file):
        jobs = [(img, g[["coords0", "coords1"]].values) for img, g in crops.groupby("img_path")]
        rows = []
        with ProcessPoolExecutor(8) as ex:
            for img, ds, shp in ex.map(image_density, jobs, chunksize=4):
                g = crops[crops.img_path == img]
                rows += [(img, r, c, d) for (r, c), d in zip(g[["coords0", "coords1"]].values, ds)]
        dens = pd.DataFrame(rows, columns=["img_path", "coords0", "coords1", "density"])
        med = dens.groupby("img_path").density.transform("median")
        dens["empty"] = dens.density <= np.maximum(0.1, 0.4 * med)
        dens["empty_abs"] = dens.density <= 0.1
        dens.to_csv(dens_file, index=False)
    dens = pd.read_csv(dens_file)
    print(f"crops {len(dens)}  NaN density {dens.density.isna().sum()}  empty (rule) {dens["empty"].mean():.3%}  "
          f"empty (abs<=0.1) {dens.empty_abs.mean():.3%}", flush=True)
    key = dens.set_index(["img_path", "coords0", "coords1"])

    summary = []
    for feat in FEATURES:
        fm = balance_meta_df(drop_nan_labels(pooled[pooled.feature == feat]), min_class_freq=0.05)
        classes = sorted(fm.feature_value.astype(str).unique()); c2i = {c: i for i, c in enumerate(classes)}
        bags, labels = build_image_bags(fm, c2i)
        # per-bag empty flags in exactly the bag-row order of build_image_bags
        flags = {"rule": [], "abs": []}
        for _img, g in fm.groupby("img_path", sort=False):
            if pd.isna(g.feature_value.iloc[0]) or c2i.get(str(g.feature_value.iloc[0])) is None: continue
            rows = pd.concat([gg for _, gg in g.groupby("embeddings_file")])
            k = key.loc[list(zip(rows.img_path, rows.coords0, rows.coords1))]
            flags["rule"].append(k["empty"].values.astype(bool)); flags["abs"].append(k.empty_abs.values.astype(bool))
        y = labels.numpy().astype(int); n = len(bags); C = len(classes)
        nf = min(10, int(np.bincount(y).min()), n)
        folds = list(StratifiedKFold(n_splits=nf, shuffle=True, random_state=42).split(np.zeros(n), y))
        ld = 1 if C <= 2 else C
        oof = {k: np.full((n, ld), np.nan, np.float32) for k in ["all", "rule", "abs"]}
        att_rows = []
        for f, (tr, va) in enumerate(folds):
            ck = sorted(glob.glob(f"{RES}/{feat}/fold_{f}/*.ckpt"), key=os.path.getmtime)[-1]
            model = AbmilLitModule.load_from_checkpoint(ck, map_location="cpu").model.eval()
            mu, sd = zscore_stats_from_bags(bags, indices=tr)
            vb = [apply_zscore_to_bags([bags[i]], mu, sd)[0] for i in va]
            with torch.no_grad():
                x, m = pad(vb); lg, a = attention(model, x, m)
                oof["all"][va] = lg.numpy().reshape(len(va), ld)
                for rule in ["rule", "abs"]:
                    em = [flags[rule][i] for i in va]
                    keep_any = np.array([(~e).any() for e in em])
                    em2 = [e if k else np.zeros_like(e) for e, k in zip(em, keep_any)]  # all-empty bag: keep as is
                    x2, m2 = pad(vb, em2); lg2, _ = attention(model, x2, m2)
                    oof[rule][va] = lg2.numpy().reshape(len(va), ld)
                am = a.mean(1).numpy()                      # [B, S] head-averaged attention
                for j, i in enumerate(va):
                    s = len(bags[i]); e = flags["rule"][i]; ea = flags["abs"][i]
                    att_rows.append(dict(bag=i, fold=f, n=s, frac_empty=e.mean(), att_empty=am[j, :s][e].sum(),
                                         frac_empty_abs=ea.mean(), att_empty_abs=am[j, :s][ea].sum(),
                                         top_is_empty=bool(e[np.argmax(am[j, :s])])))
        att = pd.DataFrame(att_rows); att.to_csv(f"{OUT}/attention_{feat}.csv", index=False)
        has = att.frac_empty > 0
        rep = pd.read_csv(f"{RUN}/results/zglobal/results.csv").set_index("feature").loc[feat, "auc"]
        row = dict(feature=feat, bags=n, classes=C, folds=nf, reported_auc=rep,
                   auc_all=_auc(y, oof["all"], C), auc_no_empty=_auc(y, oof["rule"], C), auc_no_empty_abs=_auc(y, oof["abs"], C),
                   frac_crops_empty=float(np.concatenate(flags["rule"]).mean()),
                   bags_with_empty=float(has.mean()),
                   att_mass_empty=float(att.att_empty.mean()),
                   att_mass_empty_in_bags_with_empty=float(att[has].att_empty.mean()),
                   frac_empty_in_bags_with_empty=float(att[has].frac_empty.mean()),
                   enrichment=float(att[has].att_empty.sum() / att[has].frac_empty.sum()),
                   top_crop_empty=float(att.top_is_empty.mean()),
                   att_mass_empty_abs=float(att.att_empty_abs.mean()))
        summary.append(row); print(row, flush=True)
    pd.DataFrame(summary).to_csv(f"{OUT}/summary.csv", index=False)
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
