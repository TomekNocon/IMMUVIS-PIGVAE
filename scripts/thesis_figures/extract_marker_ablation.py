"""Marker ablation (occlusion) through the full chain, Sec. 6.3 deep-dive (szary, GPU).

Raw crop (preprocessed) -> ImmuVis -> PCA-128 -> PIGVAE -> z_global -> train-fold z-score -> OOF ABMIL head.
Forward passes only (no gradients), so no IG convergence issue. Effects are oriented to the predicted
class: effect > 0 means the marker/crop SUPPORTS the model's call.

Per slide:
  ref        logit with every crop re-encoded from the raw image (consistency vs stored-bag logit)
  slide_abl  [43]     marker m zeroed in ALL crops
  crop_abl   [n, 43]  marker m zeroed in crop c only (others at reference)
  group_abl  [G], group_crop_abl [n, G]   marker groups zeroed (all crops / one crop)
  crop_occ   [n]      whole crop c zeroed (all markers)
  marker_mean [n, 43] mean preprocessed intensity per crop and marker
Cohort: 2 example slides (7064 adeno, 6577 squamous) + up to N correctly classified OOF slides per class.
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
OUT = "thesis_figures/_cache/marker_ablation"; FEAT = "DX.name"; P = 128
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
gae = load_frozen_pigvae(CKPT, "pigvae").to(dev)
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



GROUPS = {
    "epithelial/tumour": ["panCK", "K-Cadherin", "Carbonic Anhydrase", "Ki67"],
    "nuclear": ["DNA1", "DNA2", "Histone H3"],
    "immune": ["CD45", "CD3", "CD4", "CD8a", "FOXP3", "PD1", "TCF1/7", "CD20", "CD68", "CD15", "MPO", "HLADR", "Ido1"],
    "stroma/fibroblast": ["Cadherin-11", "Caveolin-1", "CD10", "CD146", "CD248", "CD73", "Collagen Type I", "FAP", "FSP1",
                          "MMP11", "MMP9", "p75", "PDGFRb", "SMA", "Vimentin", "CXCL12"],
    "vascular/lymphatic": ["CD31-vWF", "CD34", "LYVE1", "PNAd", "CCL21", "VCAM1", "Podoplanin"],
}
mlist = list(mnames); gidx = {g: [mlist.index(m) for m in ms] for g, ms in GROUPS.items()}
assert sorted(sum(gidx.values(), [])) == list(range(len(mlist))), "groups must partition the panel"
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 15)); BS = 16


@torch.no_grad()
def enc_z(X):                                     # X [n, 43, P, P] on dev -> z_global [n, D]
    return torch.cat([zglobal_of(X[s:s + BS])[0] for s in range(0, len(X), BS)])


def tiff_path(img):
    for sp in ["train", "test"]:
        q = f"{SPLIT}/{sp}/cords/imgs/{img}"
        if os.path.exists(q): return q
    return f"{TIFF}/{img}"


rng = np.random.default_rng(0)
names = [g.img_path.iloc[0] for g in groups]
sel = {names.index("7064.tiff"): "example", names.index("6577.tiff"): "example"}
for cls in [0, 1]:
    ok = np.where((y == cls) & ((oof > 0) == (cls == 1)) & (n >= 10))[0]
    ok = [i for i in ok if i not in sel]
    for i in rng.choice(ok, size=min(N_PER_CLASS, len(ok)), replace=False): sel[int(i)] = "cohort"
print(f"slides: {len(sel)}", flush=True)

for i, role in sel.items():
    f = fold_of[i]; head = heads[f]; mu, sd = stats[f]
    mu_t, sd_t = torch.from_numpy(mu).float().to(dev), torch.from_numpy(sd).float().to(dev)
    img = names[i]; r = rows_of[i]
    coords = r[["coords0", "coords1"]].values.astype(int)
    dv = dens.loc[list(zip(r.img_path, r.coords0, r.coords1))]
    pre = preprocess(tifffile.imread(tiff_path(img))).astype(np.float32)
    X = torch.from_numpy(np.stack([pre[:, a:a + P, b:b + P] for a, b in coords])).to(dev)   # [n,43,P,P]
    xb = torch.from_numpy(apply_zscore_to_bags([bags[i]], mu, sd)[0]).float()[None].to(dev)
    with torch.no_grad():
        logit_stored = head(xb)[0][0, 0].item()
        Zr = (enc_z(X) - mu_t) / sd_t                                                     # [n, D]
        logit_ref = head(Zr[None])[0][0, 0].item()
        sign = 1.0 if logit_ref > 0 else -1.0
        L = lambda bag: head(bag)[0][:, 0] * sign          # oriented logit(s) for [B, n, D]
        ref = L(Zr[None]).item()
        nC, nM = X.shape[0], X.shape[1]
        # whole-crop occlusion
        z0 = (enc_z(torch.zeros_like(X[:1])) - mu_t) / sd_t
        B = Zr[None].repeat(nC, 1, 1); B[torch.arange(nC), torch.arange(nC)] = z0[0]
        crop_occ = (ref - L(B)).cpu().numpy()
        def ablate(chans):
            Xa = X.clone(); Xa[:, chans] = 0
            Za = (enc_z(Xa) - mu_t) / sd_t
            s_eff = ref - L(Za[None]).item()
            B = Zr[None].repeat(nC, 1, 1); B[torch.arange(nC), torch.arange(nC)] = Za
            return s_eff, (ref - L(B)).cpu().numpy()
        slide_abl = np.zeros(nM); crop_abl = np.zeros((nC, nM))
        for m in range(nM):
            slide_abl[m], crop_abl[:, m] = ablate([m])
        G = list(GROUPS); group_abl = np.zeros(len(G)); group_crop_abl = np.zeros((nC, len(G)))
        for gi, g in enumerate(G):
            group_abl[gi], group_crop_abl[:, gi] = ablate(gidx[g])
    marker_mean = X.mean((2, 3)).cpu().numpy()
    top = np.argsort(-slide_abl)[:5]
    print(f"[{role}] {img} {classes[y[i]][:12]:12s} fold {f} n={nC} logit stored {logit_stored:+.2f} "
          f"re-enc {logit_ref:+.2f} | top markers " + ", ".join(f"{mlist[m]} {slide_abl[m]:+.2f}" for m in top)
          + " | groups " + ", ".join(f"{g.split('/')[0]} {v:+.2f}" for g, v in zip(G, group_abl)), flush=True)
    extra = dict(slide=pre.astype(np.float16)) if role == "example" else {}
    np.savez_compressed(f"{OUT}/{img[:-5]}.npz", img=img, label=classes[y[i]], y=y[i], role=role, fold=f,
                        logit_stored=logit_stored, logit_ref=logit_ref, coords=coords,
                        density=dv["density"].values, empty=dv["empty"].values.astype(bool),
                        crop_occ=crop_occ, slide_abl=slide_abl, crop_abl=crop_abl, group_names=np.array(G),
                        group_abl=group_abl, group_crop_abl=group_crop_abl, marker_mean=marker_mean,
                        markers=np.array(mlist), **extra)
print("done")
