"""Items 17 + 18 (Sec. 6.4 latent attribution, Sec. 6.5 faithfulness), run 81zasevr, label DX.name (szary, GPU).

All heads are the OUT-OF-FOLD ABMIL heads (10-fold, random_state=42, train-fold z-score), as in
quantify_empty_shortcut.py / extract_marker_ablation.py. Everything is oriented to the predicted class.
No ImmuVis pass: crop node features come from the stored ImmuVis h5 map -> PCA-128, which reproduces
the stored z_global (checked and printed).

Part A  decision -> latent (item 17)
  For every bag: ABMIL IG (zero baseline in z-scored space, 64 steps), NOT summed over features
  -> crop x latent-dim matrix [n, D]; per-slide dim totals; completeness check.
  Examples (7064, 6577): full matrix + for the decisive real-tissue crop the top-3 decision dims, each
  explained back onto the 16x16 node grid with encoder IG (target z_global[d]) and rollout.
Part B  faithfulness (item 18)
  B1 crop level, all bags: deletion/insertion over crops for ABMIL IG, ABMIL attention, random (5 orders).
  B2 node level, ~200 tissue crops: deletion/insertion over the 256 nodes for encoder IG, rollout,
     random (5 orders), for two targets: (t1) the crop's top decision dim z_global[d] (signed toward the
     prediction), (t2) the decision logit with that crop's z_global replaced. Baseline = zeros (PCA origin).
  Curves are normalised: (f - f(all baseline)) / (f(x) - f(all baseline)).
"""
import glob, os, sys
import h5py, joblib, numpy as np, pandas as pd, torch
from captum.attr import IntegratedGradients
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/thesis_figures")
from src.downstream.abmil.cv import zscore_stats_from_bags, apply_zscore_to_bags
from src.downstream.abmil.data import build_image_bags
from src.downstream.abmil.lit import AbmilLitModule
from src.downstream.encode import load_frozen_pigvae
from src.data.components.graphs_datamodules import DenseGraphBatch
from run_abmil import drop_nan_labels, balance_meta_df, _load_pooled_meta
from explain_rollout import attention_rollout

RUN = "/raid_encrypted/immucan/embeddings/tnocon/downstream/vae16_s0p01_film_2026-07-31"
META, RES = f"{RUN}/meta_tables/zglobal", f"{RUN}/results/zglobal/cords"
D_ = "/raid_encrypted/immucan/embeddings/tnocon/data/IMC/cords"
CKPT = "/raid_encrypted/immucan/embeddings/tnocon/logs/train/runs/2026-07-31_10-29-22/checkpoints/last.ckpt"
OUT = "thesis_figures/_cache/latent_faith"; FEAT = "DX.name"
N_NODE_CROPS = int(os.environ.get("N_NODE_CROPS", 200)); NSTEP = 16; NRAND = 5; IG_STEPS = 64
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); rng = np.random.default_rng(0)


def attention(model, x):
    v, u = torch.tanh(model.V(x)), torch.sigmoid(model.U(x))
    return torch.softmax(model.W(v * u), dim=1).transpose(1, 2)        # [B, H, n]


pooled = _load_pooled_meta(META, "cords")
fm = balance_meta_df(drop_nan_labels(pooled[pooled.feature == FEAT]), min_class_freq=0.05)
classes = sorted(fm.feature_value.astype(str).unique()); c2i = {c: i for i, c in enumerate(classes)}
bags, labels = build_image_bags(fm, c2i); y = labels.numpy()
groups = [g for _, g in fm.groupby("img_path", sort=False)]
rows_of = [pd.concat([gg for _, gg in g.groupby("embeddings_file")]) for g in groups]
names = [g.img_path.iloc[0] for g in groups]
dens = pd.read_csv("thesis_figures/_cache/empty_shortcut/crop_density.csv").set_index(["img_path", "coords0", "coords1"])
empty = [dens.loc[list(zip(r.img_path, r.coords0, r.coords1))]["empty"].values.astype(bool) for r in rows_of]
folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(np.zeros(len(bags)), y))
fold_of = np.empty(len(bags), int)
for f, (_, va) in enumerate(folds): fold_of[va] = f
heads, stats = {}, {}
for f, (tr, va) in enumerate(folds):
    ck = sorted(glob.glob(f"{RES}/{FEAT}/fold_{f}/*.ckpt"), key=os.path.getmtime)[-1]
    heads[f] = AbmilLitModule.load_from_checkpoint(ck, map_location="cpu").model.eval().to(dev)
    heads[f].requires_grad_(False)
    stats[f] = tuple(torch.from_numpy(s).float().to(dev) for s in zscore_stats_from_bags(bags, indices=tr))
Dz = bags[0].shape[1]
print(f"{len(bags)} bags, classes {classes}, z_global dim {Dz}", flush=True)

gae = load_frozen_pigvae(CKPT, "pigvae").to(dev).eval(); gae.requires_grad_(False)
pca = joblib.load(f"{D_}/pca_model_128_center_crop_16.pkl")
V = torch.tensor(pca.components_, dtype=torch.float32, device=dev); pm = torch.tensor(pca.mean_, dtype=torch.float32, device=dev)
h5 = {s: h5py.File(f"{D_}/{s}.h5", "r") for s in ["train", "test"]}
h5paths = {s: np.array([p.decode().split("/")[-1] for p in h5[s]["paths"][:]]) for s in h5}


def bag_of(i):
    mu, sd = stats[fold_of[i]]
    return ((torch.from_numpy(np.asarray(bags[i])).float().to(dev) - mu) / sd)[None]      # [1, n, D]


def zg(nf):                                         # [B, 256, F] -> z_global [B, D]
    b = DenseGraphBatch(node_features=nf, edge_features=torch.empty(0, device=dev),
                        mask=torch.ones(nf.shape[0], nf.shape[1], dtype=torch.bool, device=dev))
    return gae.encode(b, sample=False)[1]


def nodes_of(img, a0, a1):                         # stored ImmuVis map of one crop -> PCA node features [256, F]
    s = "train" if (h5paths["train"] == img).any() else "test"
    rows = np.where(h5paths[s] == img)[0]; pos = h5[s]["positions"][rows[0]:rows[-1] + 1]
    hit = rows[[j for j, p in enumerate(pos) if int(p[0]) == a0 and int(p[1]) == a1]][0]
    lat = torch.from_numpy(h5[s]["embeddings"][hit]).float().to(dev)                    # [768, 16, 16]
    return (lat.reshape(768, 256).T - pm) @ V.T


def norm_curve(c, full, base):
    return (c - base) / (full - base) if abs(full - base) > 1e-6 else np.full_like(c, np.nan)


def auc(c):
    return float(np.trapz(c, dx=1.0 / (len(c) - 1)))


# ---------------------------------------------------------------- Part A + B1: every bag
cohort = dict(img=[], y=[], logit=[], dimtot=[], dimabs=[], comp_err=[], b1=[])
examples = {}
for i in range(len(bags)):
    f = fold_of[i]; head = heads[f]; xb = bag_of(i); n = xb.shape[1]
    with torch.no_grad():
        logit = head(xb)[0][0, 0].item(); att = attention(head, xb).mean(1)[0].cpu().numpy()
    sign = 1.0 if logit > 0 else -1.0
    F_ = lambda b: head(b)[0][:, 0] * sign
    attr, delta = IntegratedGradients(F_).attribute(xb, baselines=torch.zeros_like(xb), n_steps=IG_STEPS,
                                                    return_convergence_delta=True)
    M = attr[0].detach().cpu().numpy()                                                   # [n, D]
    with torch.no_grad():
        fx, f0 = F_(xb).item(), F_(torch.zeros_like(xb)).item()
    cohort["img"].append(names[i]); cohort["y"].append(y[i]); cohort["logit"].append(logit)
    cohort["dimtot"].append(M.sum(0)); cohort["dimabs"].append(np.abs(M).sum(0))
    cohort["comp_err"].append(abs(M.sum() - (fx - f0)) / max(abs(fx - f0), 1e-8))
    # B1: crop-level deletion / insertion, per-crop granularity, batched
    ig_c = M.sum(1)
    orders = {"ig": np.argsort(-ig_c), "attention": np.argsort(-att)}
    orders.update({f"random{r}": rng.permutation(n) for r in range(NRAND)})
    res = {}
    with torch.no_grad():
        for meth, o in orders.items():
            for mode in ["deletion", "insertion"]:
                B = (xb.repeat(n + 1, 1, 1) if mode == "deletion" else torch.zeros_like(xb).repeat(n + 1, 1, 1))
                src = torch.zeros_like(xb[0]) if mode == "deletion" else xb[0]
                for k in range(1, n + 1):
                    ok_ = torch.as_tensor(o[:k], device=dev); B[k, ok_] = src[ok_]
                res[(meth, mode)] = norm_curve(F_(B).cpu().numpy(), fx, f0)
    rec = {}
    for meth in ["ig", "attention", "random"]:
        for mode in ["deletion", "insertion"]:
            if meth == "random":
                rec[(meth, mode)] = float(np.mean([auc(res[(f"random{r}", mode)]) for r in range(NRAND)]))
            else:
                rec[(meth, mode)] = auc(res[(meth, mode)])
    cohort["b1"].append([rec[(m, md)] for m in ["ig", "attention", "random"] for md in ["deletion", "insertion"]])
    if names[i] in ("7064.tiff", "6577.tiff"):
        examples[names[i]] = dict(i=i, M=M, att=att, logit=logit, sign=sign,
                                  curves={f"{m}_{md}": res[(m, md)] for m in ["ig", "attention", "random0"]
                                          for md in ["deletion", "insertion"]})
    if i % 50 == 0:
        print(f"  bag {i}/{len(bags)} {names[i]} logit {logit:+.2f} IG compl. err {cohort['comp_err'][-1]:.3f} "
              f"| B1 del AUC ig/att/rand {rec[('ig','deletion')]:.3f}/{rec[('attention','deletion')]:.3f}/"
              f"{rec[('random','deletion')]:.3f}", flush=True)

np.savez_compressed(f"{OUT}/cohort.npz", img=np.array(cohort["img"]), y=np.array(cohort["y"]),
                    logit=np.array(cohort["logit"]), dimtot=np.array(cohort["dimtot"]),
                    dimabs=np.array(cohort["dimabs"]), comp_err=np.array(cohort["comp_err"]),
                    b1=np.array(cohort["b1"]), b1_cols=np.array([f"{m}_{md}" for m in ["ig", "attention", "random"]
                                                                  for md in ["deletion", "insertion"]]),
                    classes=np.array(classes))
b1 = np.array(cohort["b1"]); ok = ~np.isnan(b1).any(1)
print("B1 crop-level mean AUC (del, ins):", {c: round(float(v), 3) for c, v in
      zip(["ig_del", "ig_ins", "att_del", "att_ins", "rnd_del", "rnd_ins"], b1[ok].mean(0))},
      f"| IG completeness median rel err {np.median(cohort['comp_err']):.4f}", flush=True)


# ---------------------------------------------------------------- node-level helpers
def node_target_fns(i, k, d, sign):
    """t1: z_global[d] signed toward the prediction (sign of its decision IG); t2: oriented decision logit."""
    f = fold_of[i]; head = heads[f]; mu, sd = stats[f]; xb = bag_of(i)

    def t2(nf):
        zn = (zg(nf) - mu) / sd
        b = xb.expand(nf.shape[0], -1, -1).clone(); b[:, k] = zn
        return head(b)[0][:, 0] * sign
    return t2


def node_curves(fn, x, imp_orders, chunk=48):
    """x [256, F]; returns {name: (del_curve, ins_curve)} at NSTEP+1 points, raw (not normalised)."""
    ks = [round(j * 256 / NSTEP) for j in range(NSTEP + 1)]
    sets, keys = [], []
    for name, o in imp_orders.items():
        o = torch.as_tensor(o, device=dev)
        for mode in ["deletion", "insertion"]:
            for kk in ks:
                cur = x.clone() if mode == "deletion" else torch.zeros_like(x)
                cur[o[:kk]] = 0 if mode == "deletion" else x[o[:kk]]
                sets.append(cur); keys.append((name, mode))
    X = torch.stack(sets)
    with torch.no_grad():
        out = torch.cat([fn(X[s:s + chunk]) for s in range(0, len(X), chunk)]).cpu().numpy()
    res = {}
    for j, key in enumerate(keys):
        res.setdefault(key, []).append(out[j])
    return {k_: np.array(v) for k_, v in res.items()}


def explain_crop(i, k, d, sign_d, sign):
    """Encoder IG + rollout maps for crop k of bag i, and node-level curves for t1 and t2."""
    r = rows_of[i]; a0, a1 = int(r.coords0.iloc[k]), int(r.coords1.iloc[k])
    x = nodes_of(names[i], a0, a1)
    with torch.no_grad():
        z_re = zg(x[None])[0]
    z_st = torch.from_numpy(np.asarray(bags[i])[k]).float().to(dev)
    zerr = (z_re - z_st).abs().max().item()
    t1 = lambda nf: zg(nf)[:, d] * sign_d
    t2 = node_target_fns(i, k, d, sign)
    out = dict(zerr=zerr, coords=(a0, a1))
    roll = attention_rollout(gae, x[None])[0].cpu().numpy()
    out["rollout"] = roll
    for tname, fn in [("t1", t1), ("t2", t2)]:
        ig, dl = IntegratedGradients(fn).attribute(x[None], baselines=torch.zeros_like(x[None]), n_steps=IG_STEPS,
                                                   internal_batch_size=32, return_convergence_delta=True)
        ig_n = ig[0].sum(-1).detach().cpu().numpy()                                       # [256] signed
        with torch.no_grad():
            fx, f0 = fn(x[None]).item(), fn(torch.zeros_like(x[None])).item()
        orders = {"ig": np.argsort(-ig_n), "rollout": np.argsort(-roll)}
        orders.update({f"random{q}": rng.permutation(256) for q in range(NRAND)})
        cur = node_curves(fn, x, orders)
        out[tname] = dict(ig=ig_n, fx=fx, f0=f0, comp_err=abs(ig_n.sum() - (fx - f0)) / max(abs(fx - f0), 1e-8),
                          curves={f"{m}_{md}": norm_curve(cur[(m, md)], fx, f0)
                                  for m in orders for md in ["deletion", "insertion"]})
    return out


# ---------------------------------------------------------------- Part A examples: decisive crop, top-3 dims
for img, ex in examples.items():
    i = ex["i"]; M = ex["M"]; e = empty[i]
    k = int(np.argmax(np.where(~e, M.sum(1), -np.inf)))
    top_d = np.argsort(-M[k])[:3]
    per_dim = {}
    for d in top_d:
        sd_ = 1.0 if M[k, d] >= 0 else -1.0
        o = explain_crop(i, k, int(d), sd_, ex["sign"])
        per_dim[int(d)] = o
        print(f"[example {img}] crop {k} {o['coords']} dim {d} IG {M[k, d]:+.3f} | z re-encode maxabs {o['zerr']:.2e} "
              f"| t1 compl err {o['t1']['comp_err']:.3f} t2 {o['t2']['comp_err']:.3f} "
              f"| corr(IG t1, rollout) {np.corrcoef(o['t1']['ig'], o['rollout'])[0,1]:+.2f}", flush=True)
    np.savez_compressed(f"{OUT}/example_{img[:-5]}.npz", img=img, M=M, att=ex["att"], logit=ex["logit"],
                        empty=e, coords=rows_of[i][["coords0", "coords1"]].values.astype(int), decisive=k,
                        top_dims=top_d, M_dec=M[k],
                        ig_t1=np.array([per_dim[int(d)]["t1"]["ig"] for d in top_d]),
                        ig_t2=per_dim[int(top_d[0])]["t2"]["ig"],
                        rollout=per_dim[int(top_d[0])]["rollout"],
                        **{f"crop_curve_{k_}": v for k_, v in ex["curves"].items()})

# ---------------------------------------------------------------- Part B2: ~200 tissue crops
cand = [(i, k) for i in range(len(bags)) for k in np.where(~empty[i])[0]]
by_cls = {c: [t for t in cand if y[t[0]] == c] for c in [0, 1]}
pick = []
for c in [0, 1]:
    idx = rng.choice(len(by_cls[c]), size=min(N_NODE_CROPS // 2, len(by_cls[c])), replace=False)
    pick += [by_cls[c][j] for j in idx]
ex_by_i = {}
recs = []
for q, (i, k) in enumerate(pick):
    if i not in ex_by_i:
        f = fold_of[i]; xb = bag_of(i)
        with torch.no_grad(): lg = heads[f](xb)[0][0, 0].item()
        sg = 1.0 if lg > 0 else -1.0
        A = IntegratedGradients(lambda b: heads[f](b)[0][:, 0] * sg).attribute(
            xb, baselines=torch.zeros_like(xb), n_steps=IG_STEPS)[0].detach().cpu().numpy()
        ex_by_i[i] = (A, sg, lg)
    A, sg, lg = ex_by_i[i]
    d = int(np.argmax(np.abs(A[k]))); sd_ = 1.0 if A[k, d] >= 0 else -1.0
    o = explain_crop(i, k, d, sd_, sg)
    rec = dict(img=names[i], y=y[i], crop=k, dim=d, logit=lg, zerr=o["zerr"],
               corr_ig_roll_t1=np.corrcoef(o["t1"]["ig"], o["rollout"])[0, 1],
               corr_ig_roll_t2=np.corrcoef(o["t2"]["ig"], o["rollout"])[0, 1])
    for t in ["t1", "t2"]:
        rec[f"{t}_comp_err"] = o[t]["comp_err"]; rec[f"{t}_dfx"] = o[t]["fx"] - o[t]["f0"]
        for m in ["ig", "rollout"]:
            for md in ["deletion", "insertion"]:
                c = o[t]["curves"][f"{m}_{md}"]; rec[f"{t}_{m}_{md}"] = auc(c); rec[f"{t}_{m}_{md}_curve"] = c
        for md in ["deletion", "insertion"]:
            cs = np.array([o[t]["curves"][f"random{r}_{md}"] for r in range(NRAND)])
            rec[f"{t}_random_{md}"] = float(np.mean([auc(c) for c in cs])); rec[f"{t}_random_{md}_curve"] = cs.mean(0)
    recs.append(rec)
    if q % 20 == 0:
        print(f"  B2 {q}/{len(pick)} {names[i]} crop {k} dim {d} zerr {o['zerr']:.1e} | "
              + " | ".join(f"{t} del ig/roll/rnd {rec[f'{t}_ig_deletion']:.2f}/{rec[f'{t}_rollout_deletion']:.2f}/"
                           f"{rec[f'{t}_random_deletion']:.2f} compl {rec[f'{t}_comp_err']:.2f}" for t in ["t1", "t2"]),
              flush=True)

scal = {k_: np.array([r[k_] for r in recs]) for k_ in recs[0] if not k_.endswith("_curve")}
curv = {k_: np.stack([r[k_] for r in recs]) for k_ in recs[0] if k_.endswith("_curve")}
np.savez_compressed(f"{OUT}/faith_nodes.npz", **scal, **curv)
for t in ["t1", "t2"]:
    ok = np.isfinite(scal[f"{t}_ig_deletion"]) & (np.abs(scal[f"{t}_dfx"]) > 1e-3)
    print(f"B2 {t} (n={ok.sum()}): " + ", ".join(
        f"{m} del {np.nanmean(scal[f'{t}_{m}_deletion'][ok]):.3f} ins {np.nanmean(scal[f'{t}_{m}_insertion'][ok]):.3f}"
        for m in ["ig", "rollout", "random"]) + f" | median IG compl err {np.median(scal[f'{t}_comp_err']):.3f}", flush=True)
print("done")
