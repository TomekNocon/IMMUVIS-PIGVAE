"""Spatial attribution examples for Sec. 6.3 (szary, GPU), reference run 81zasevr, label DX.name.

For one correctly classified slide per class, using the OUT-OF-FOLD head (the fold that held the
slide out, reproduced exactly as in quantify_empty_shortcut.py):
  - per-crop ABMIL decision attribution: Integrated Gradients on the bag (baseline = zeros in the
    train-fold z-scored space, 50 steps), summed over features, oriented to the predicted class;
    plus the head-averaged attention weights; plus tissue density / empty flag per crop;
  - the preprocessed slide (arcsinh -> Butterworth -> clip/4, as fed to ImmuVis), for rendering.
End-to-end raw-image attribution for the strongest real-tissue crop of each slide: Integrated
Gradients on the preprocessed 43x128x128 crop through ImmuVis -> PCA-128 -> PIGVAE -> z_global ->
(train-fold z-score) -> ABMIL decision logit, other crops of the bag held fixed (baseline = zeros).
Consistency checks: ImmuVis(crop) vs stored h5 embedding, chain logit vs head logit.
"""
import os, sys
import h5py, joblib, numpy as np, pandas as pd, tifffile, torch
from captum.attr import IntegratedGradients
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/thesis_figures")
import glob
from src.downstream.abmil.cv import zscore_stats_from_bags, apply_zscore_to_bags
from src.downstream.abmil.data import build_image_bags
from src.downstream.abmil.lit import AbmilLitModule
from src.downstream.encode import load_frozen_pigvae
from src.data.components.graphs_datamodules import DenseGraphBatch
from run_abmil import drop_nan_labels, balance_meta_df, _load_pooled_meta
from fig_cords_crop import preprocess
import immuvis_encoder as ie

RUN = "/raid_encrypted/immucan/embeddings/tnocon/downstream/vae16_s0p01_film_2026-07-31"
META, RES = f"{RUN}/meta_tables/zglobal", f"{RUN}/results/zglobal/cords"
TIFF = "/raid_encrypted/immucan/IMC/immu-vis-other-ds/full-20122025"
SPLIT = "/raid_encrypted/immucan/immuvis_split"
D = "/raid_encrypted/immucan/embeddings/tnocon/data/IMC/cords"
CKPT = "/raid_encrypted/immucan/embeddings/tnocon/logs/train/runs/2026-07-31_10-29-22/checkpoints/last.ckpt"
OUT = "thesis_figures/_cache/spatial_v3"; FEAT = "DX.name"; P = 128
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)


def attention(model, x, mask=None):
    v, u = torch.tanh(model.V(x)), torch.sigmoid(model.U(x))
    a = model.W(v * u)
    if mask is not None: a = a.masked_fill(mask.unsqueeze(2), -1e9)
    a = torch.softmax(a, dim=1).transpose(1, 2)
    return model.classifier(torch.bmm(a, x).reshape(x.size(0), -1)), a


pooled = _load_pooled_meta(META, "cords")
fm = balance_meta_df(drop_nan_labels(pooled[pooled.feature == FEAT]), min_class_freq=0.05)
classes = sorted(fm.feature_value.astype(str).unique()); c2i = {c: i for i, c in enumerate(classes)}
bags, labels = build_image_bags(fm, c2i); y = labels.numpy()
groups = [g for _, g in fm.groupby("img_path", sort=False)]
rows_of = [pd.concat([gg for _, gg in g.groupby("embeddings_file")]) for g in groups]
dens = pd.read_csv("thesis_figures/_cache/empty_shortcut/crop_density.csv").set_index(["img_path", "coords0", "coords1"])
folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(np.zeros(len(bags)), y))
fold_of = np.empty(len(bags), int)
for f, (_, va) in enumerate(folds): fold_of[va] = f

# OOF logits for every bag, to choose representative slides
heads, stats, oof = {}, {}, np.zeros(len(bags))
for f, (tr, va) in enumerate(folds):
    ck = sorted(glob.glob(f"{RES}/{FEAT}/fold_{f}/*.ckpt"), key=os.path.getmtime)[-1]
    heads[f] = AbmilLitModule.load_from_checkpoint(ck, map_location="cpu").model.eval().to(dev)
    stats[f] = zscore_stats_from_bags(bags, indices=tr)
    with torch.no_grad():
        for i in va:
            xb = torch.from_numpy(apply_zscore_to_bags([bags[i]], *stats[f])[0]).float()[None].to(dev)
            oof[i] = heads[f](xb)[0].item()
empty = [dens.loc[list(zip(r.img_path, r.coords0, r.coords1))]["empty"].values.astype(bool) for r in rows_of]
n = np.array([len(b) for b in bags]); ef = np.array([e.mean() for e in empty])
pick = []
for cls in [0, 1]:
    ok = (y == cls) & ((oof > 0) == (cls == 1)) & (n >= 25) & (n <= 42) & (ef <= 0.2)
    idx = np.where(ok)[0]; pick.append(int(idx[np.argmax(np.abs(oof[idx]))]))
print("classes", classes, "picked", [(groups[i].img_path.iloc[0], classes[y[i]], round(oof[i], 2), n[i]) for i in pick], flush=True)

imm, (miss, unexp) = ie.load_immuvis_refactored(device=dev); imm.requires_grad_(False)
print("ImmuVis load missing/unexpected:", len(miss), len(unexp), flush=True)
cids, mnames = ie.dataset_channel_ids("cords"); cids = cids.to(dev)
gae = load_frozen_pigvae(CKPT, "vae16_fb0p0_film").to(dev)
pca = joblib.load(f"{D}/pca_model_128_center_crop_16.pkl")
V = torch.tensor(pca.components_, dtype=torch.float32, device=dev); pm = torch.tensor(pca.mean_, dtype=torch.float32, device=dev)
h5 = {s: h5py.File(f"{D}/{s}.h5", "r") for s in ["train", "test"]}
h5paths = {s: np.array([p.decode().split("/")[-1] for p in h5[s]["paths"][:]]) for s in h5}


def zglobal_of(x):                                   # x [B, 43, 128, 128] preprocessed crops
    lat = ie.encode(imm, x, cids[None].expand(x.shape[0], -1))      # [B, 768, 16, 16]
    nodes = lat.reshape(x.shape[0], 768, 256).transpose(1, 2)
    nf = (nodes - pm) @ V.T
    batch = DenseGraphBatch(node_features=nf, edge_features=torch.empty(0, device=dev),
                            mask=torch.ones(x.shape[0], 256, dtype=torch.bool, device=dev))
    return gae.encode(batch, sample=False)[1], lat


for tag, i in zip(["adeno", "squam"], pick):
    f = fold_of[i]; head = heads[f]; mu, sd = stats[f]
    img = groups[i].img_path.iloc[0]; r = rows_of[i]
    xb = torch.from_numpy(apply_zscore_to_bags([bags[i]], mu, sd)[0]).float()[None].to(dev)
    with torch.no_grad():
        logit, att = attention(head, xb)
    sign = 1.0 if logit.item() > 0 else -1.0
    ig = IntegratedGradients(lambda inp: head(inp)[0][:, 0] * sign)
    attr = ig.attribute(xb, baselines=torch.zeros_like(xb), n_steps=50).sum(-1)[0].cpu().numpy()
    coords = r[["coords0", "coords1"]].values.astype(int)
    dvals = dens.loc[list(zip(r.img_path, r.coords0, r.coords1))]
    e = dvals["empty"].values.astype(bool)
    k = int(np.argmax(np.where(~e, attr, -np.inf)))                       # strongest real-tissue crop
    a0, a1 = coords[k]
    s = "train" if (h5paths["train"] == img).any() else "test"
    rows = np.where(h5paths[s] == img)[0]; pos = h5[s]["positions"][rows[0]:rows[-1] + 1]
    hit = rows[[j for j, p in enumerate(pos) if int(p[0]) == a0 and int(p[1]) == a1]][0]
    stored = torch.from_numpy(h5[s]["embeddings"][hit]).to(dev).float()
    # check 1 (diagnostic): which TIFF source / coordinate orientation reproduces the stored embedding?
    srcs = {"full": f"{TIFF}/{img}"}
    for sp in ["train", "test"]:
        q = f"{SPLIT}/{sp}/cords/imgs/{img}"
        if os.path.exists(q): srcs["split_" + sp] = q
    best = None
    for name, path in srcs.items():
        pre_v = preprocess(tifffile.imread(path)).astype(np.float32)
        for orient, (r0_, c0_) in {"rc": (a0, a1), "cr": (a1, a0)}.items():
            if r0_ + P > pre_v.shape[1] or c0_ + P > pre_v.shape[2]: continue
            cr_ = torch.from_numpy(pre_v[:, r0_:r0_ + P, c0_:c0_ + P].copy())[None].to(dev)
            with torch.no_grad():
                lat_ = ie.encode(imm, cr_, cids[None])
            c = torch.corrcoef(torch.stack([lat_[0].flatten().float(), stored.flatten()]))[0, 1].item()
            mad = (lat_[0].float() - stored).abs().max().item()
            print(f"      diag src={name:11s} orient={orient} shape={pre_v.shape} corr={c:.5f} maxabs={mad:.4g}", flush=True)
            if best is None or c > best[0]: best = (c, name, orient, pre_v, r0_, c0_)
    c1, src_name, orient, pre, r0, c0 = best
    crop = torch.from_numpy(pre[:, r0:r0 + P, c0:c0 + P].copy())[None].to(dev)
    print(f"      using src={src_name} orient={orient} corr={c1:.5f}", flush=True)
    # check 2: chain logit (crop k replaced by its re-encoded z_global) vs head logit on the stored bag
    def chain(x):
        z, _ = zglobal_of(x)
        zn = (z - torch.from_numpy(mu).float().to(dev)) / torch.from_numpy(sd).float().to(dev)
        bag = xb.expand(x.shape[0], -1, -1).clone(); bag[:, k] = zn
        return head(bag)[0][:, 0] * sign
    with torch.no_grad():
        c2 = (chain(crop).item() * sign, logit.item())
    print(f"[{tag}] {img} fold {f} logit {logit.item():+.3f} | crop {k} at ({r0},{c0}) | "
          f"check1 corr(ImmuVis, stored)={c1:.5f} | check2 chain logit {c2[0]:+.4f} vs head {c2[1]:+.4f}", flush=True)
    with torch.no_grad():
        delta = chain(crop).item() - chain(torch.zeros_like(crop)).item()
    comp = {}
    for ns in [64, 128, 256]:
        pix_ = IntegratedGradients(chain).attribute(crop, baselines=torch.zeros_like(crop), n_steps=ns,
                                                    internal_batch_size=8)[0].detach().cpu().numpy()
        comp[ns] = float(pix_.sum())
        print(f"      IG n_steps={ns:3d}: sum(attr) {comp[ns]:+.4f} vs f(x)-f(0) {delta:+.4f} "
              f"(rel err {abs(comp[ns] - delta) / max(abs(delta), 1e-8):.2f})", flush=True)
        pix = pix_
    np.savez_compressed(f"{OUT}/{tag}_{img[:-5]}.npz", img=img, label=classes[y[i]], logit=logit.item(), fold=f,
                        coords=coords, ig=attr * 1.0, att=att.mean(1)[0].cpu().numpy(), density=dvals["density"].values,
                        empty=e, decisive=k, slide=pre.astype(np.float16), crop_attr=pix.astype(np.float32),
                        markers=np.array(mnames), check_corr=c1, check_logit=np.array(c2), src=src_name, orient=orient,
                        crop_rc=np.array([r0, c0]), delta=delta, completeness=np.array([[n_, v_] for n_, v_ in comp.items()]))
print("done")
