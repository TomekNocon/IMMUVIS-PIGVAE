"""Decision -> latent attribution (item 17, Sec. 6.4) and faithfulness (item 18, Sec. 6.5) (szary, GPU).

All heads are the out-of-fold ABMIL heads of the reference run, label DX.name; everything is oriented
to the predicted class. No ImmuVis pass: crop node features come from the stored ImmuVis map ->
PCA-128, which reproduces the stored z_global (checked and printed).

Part A  decision -> latent: for every bag, ABMIL IG (zero baseline in z-scored space, 64 steps) NOT
        summed over features -> crop x latent-dim matrix, per-slide dim totals, completeness. For the
        examples (7064, 6577): the decisive real-tissue crop's top-3 decision dims, each explained back
        onto the 16x16 node grid with encoder IG (target z_global[d]) and attention rollout.
Part B  faithfulness:
        B1 crop level, every bag: deletion / insertion over crops for ABMIL IG, attention, random (5 orders).
        B2 node level, ~200 tissue crops: deletion / insertion over the 256 nodes for encoder IG,
           rollout, random (5 orders), for (t1) the crop's top decision dim (signed toward the
           prediction) and (t2) the decision logit with that crop's z_global replaced (zero baseline).
Curves are normalised: (f - f(all baseline)) / (f(x) - f(all baseline)).
-> ``figures/_cache/latent_faith/{cohort,faith_nodes,example_<image>}.npz``

Usage:
    uv run python scripts/thesis_figures/extract_latent_faithfulness.py --node-crops 200
"""

import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import numpy as np
import torch
from captum.attr import IntegratedGradients

from figlib import abmil, paths
from figlib.chain import ReferenceChain
from src.explainability.rollout import attention_rollout

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXAMPLES = ("7064.tiff", "6577.tiff")
IG_STEPS = 64
N_RANDOM = 5  # random orders per curve
NODE_STEPS = 16  # node curves are evaluated at 17 points
MODES = ("deletion", "insertion")
CROP_METHODS = ("ig", "attention", "random")


def normalised(curve, full: float, base: float) -> np.ndarray:
    return (
        (curve - base) / (full - base) if abs(full - base) > 1e-6 else np.full_like(curve, np.nan)
    )


def auc(curve) -> float:
    return float(np.trapezoid(curve, dx=1.0 / (len(curve) - 1)))


class Analysis:
    def __init__(self, n_node_crops: int):
        torch.manual_seed(0)
        self.rng = np.random.default_rng(0)
        self.n_node_crops = n_node_crops
        self.cohort = abmil.load_cohort("DX.name")
        self.models = abmil.load_fold_models(self.cohort, DEVICE)
        self.stats = {
            f: tuple(torch.from_numpy(s).float().to(DEVICE) for s in st)
            for f, st in self.models.stats.items()
        }
        self.chain = ReferenceChain(DEVICE)
        print(
            f"{len(self.cohort.bags)} bags, classes {self.cohort.classes}, z_global dim "
            f"{self.cohort.bags[0].shape[1]}",
            flush=True,
        )

    def bag(self, i: int) -> torch.Tensor:
        return self.models.zscored_bag(self.cohort, i, DEVICE)

    def head(self, i: int):
        return self.models.heads[self.cohort.fold_of[i]]

    # ---- Part A + B1: every bag --------------------------------------------------------------------
    def crop_curves(self, decision, bag, fx: float, f0: float, orders: dict) -> dict:
        """Normalised crop-level deletion / insertion curves, batched over the n+1 steps."""
        n = bag.shape[1]
        curves = {}
        with torch.no_grad():
            for method, order in orders.items():
                for mode in MODES:
                    steps = (bag if mode == "deletion" else torch.zeros_like(bag)).repeat(
                        n + 1, 1, 1
                    )
                    source = torch.zeros_like(bag[0]) if mode == "deletion" else bag[0]
                    for k in range(1, n + 1):
                        idx = torch.as_tensor(order[:k], device=DEVICE)
                        steps[k, idx] = source[idx]
                    curves[(method, mode)] = normalised(decision(steps).cpu().numpy(), fx, f0)
        return curves

    def every_bag(self):
        cohort = {
            "img": [],
            "y": [],
            "logit": [],
            "dimtot": [],
            "dimabs": [],
            "comp_err": [],
            "b1": [],
        }
        examples = {}
        names, y = self.cohort.names, self.cohort.y
        for i in range(len(self.cohort.bags)):
            head, bag = self.head(i), self.bag(i)
            n = bag.shape[1]
            with torch.no_grad():
                logit = head(bag)[0][0, 0].item()
                att = abmil.gated_attention(head, bag)[1].mean(1)[0].cpu().numpy()
            sign = 1.0 if logit > 0 else -1.0

            def decision(b, head=head, sign=sign):
                return head(b)[0][:, 0] * sign

            attr = IntegratedGradients(decision).attribute(
                bag,
                baselines=torch.zeros_like(bag),
                n_steps=IG_STEPS,
                return_convergence_delta=True,
            )[0]
            dims = attr[0].detach().cpu().numpy()  # [n, D]
            with torch.no_grad():
                fx, f0 = decision(bag).item(), decision(torch.zeros_like(bag)).item()
            cohort["img"].append(names[i])
            cohort["y"].append(y[i])
            cohort["logit"].append(logit)
            cohort["dimtot"].append(dims.sum(0))
            cohort["dimabs"].append(np.abs(dims).sum(0))
            cohort["comp_err"].append(abs(dims.sum() - (fx - f0)) / max(abs(fx - f0), 1e-8))

            orders = {"ig": np.argsort(-dims.sum(1)), "attention": np.argsort(-att)}
            orders.update({f"random{r}": self.rng.permutation(n) for r in range(N_RANDOM)})
            curves = self.crop_curves(decision, bag, fx, f0, orders)
            aucs = {}
            for method in CROP_METHODS:
                for mode in MODES:
                    aucs[(method, mode)] = (
                        float(np.mean([auc(curves[(f"random{r}", mode)]) for r in range(N_RANDOM)]))
                        if method == "random"
                        else auc(curves[(method, mode)])
                    )
            cohort["b1"].append([aucs[(m, mode)] for m in CROP_METHODS for mode in MODES])
            if names[i] in EXAMPLES:
                examples[names[i]] = {
                    "i": i,
                    "M": dims,
                    "att": att,
                    "logit": logit,
                    "sign": sign,
                    "curves": {
                        f"{m}_{mode}": curves[(m, mode)]
                        for m in ["ig", "attention", "random0"]
                        for mode in MODES
                    },
                }
            if i % 50 == 0:
                print(
                    f"  bag {i}/{len(self.cohort.bags)} {names[i]} logit {logit:+.2f} IG compl. err "
                    f"{cohort['comp_err'][-1]:.3f} | B1 del AUC ig/att/rand {aucs[('ig', 'deletion')]:.3f}/"
                    f"{aucs[('attention', 'deletion')]:.3f}/{aucs[('random', 'deletion')]:.3f}",
                    flush=True,
                )
        return cohort, examples

    # ---- node level --------------------------------------------------------------------------------
    def node_curves(self, target, x: torch.Tensor, orders: dict, chunk: int = 48) -> dict:
        """Raw node-level deletion / insertion curves at NODE_STEPS + 1 points for every order."""
        ks = [round(j * 256 / NODE_STEPS) for j in range(NODE_STEPS + 1)]
        inputs, keys = [], []
        for name, order in orders.items():
            order = torch.as_tensor(order, device=DEVICE)
            for mode in MODES:
                for k in ks:
                    cur = x.clone() if mode == "deletion" else torch.zeros_like(x)
                    cur[order[:k]] = 0 if mode == "deletion" else x[order[:k]]
                    inputs.append(cur)
                    keys.append((name, mode))
        stacked = torch.stack(inputs)
        with torch.no_grad():
            out = (
                torch
                .cat([target(stacked[s : s + chunk]) for s in range(0, len(stacked), chunk)])
                .cpu()
                .numpy()
            )
        curves = {}
        for j, key in enumerate(keys):
            curves.setdefault(key, []).append(out[j])
        return {key: np.array(v) for key, v in curves.items()}

    def explain_crop(self, i: int, k: int, d: int, sign_d: float, sign: float) -> dict:
        """Encoder IG + rollout for crop k of bag i, and node-level curves for both targets."""
        rows = self.cohort.rows[i]
        a0, a1 = int(rows.coords0.iloc[k]), int(rows.coords1.iloc[k])
        x = self.chain.stored_nodes(self.cohort.names[i], a0, a1)
        with torch.no_grad():
            z_re = self.chain.z_global(x[None])[0]
        z_stored = torch.from_numpy(np.asarray(self.cohort.bags[i])[k]).float().to(DEVICE)
        head, bag = self.head(i), self.bag(i)
        mu, sd = self.stats[self.cohort.fold_of[i]]

        def t1(nf):  # the decision dimension, signed toward the prediction
            return self.chain.z_global(nf)[:, d] * sign_d

        def t2(nf):  # the decision logit with crop k replaced
            b = bag.expand(nf.shape[0], -1, -1).clone()
            b[:, k] = (self.chain.z_global(nf) - mu) / sd
            return head(b)[0][:, 0] * sign

        rollout = attention_rollout(self.chain.gae, x[None])[0].cpu().numpy()
        out = {"zerr": (z_re - z_stored).abs().max().item(), "coords": (a0, a1), "rollout": rollout}
        for name, target in [("t1", t1), ("t2", t2)]:
            ig = IntegratedGradients(target).attribute(
                x[None],
                baselines=torch.zeros_like(x[None]),
                n_steps=IG_STEPS,
                internal_batch_size=32,
                return_convergence_delta=True,
            )[0]
            ig_nodes = ig[0].sum(-1).detach().cpu().numpy()  # [256], signed
            with torch.no_grad():
                fx, f0 = target(x[None]).item(), target(torch.zeros_like(x[None])).item()
            orders = {"ig": np.argsort(-ig_nodes), "rollout": np.argsort(-rollout)}
            orders.update({f"random{q}": self.rng.permutation(256) for q in range(N_RANDOM)})
            raw = self.node_curves(target, x, orders)
            out[name] = {
                "ig": ig_nodes,
                "fx": fx,
                "f0": f0,
                "comp_err": abs(ig_nodes.sum() - (fx - f0)) / max(abs(fx - f0), 1e-8),
                "curves": {
                    f"{m}_{mode}": normalised(raw[(m, mode)], fx, f0)
                    for m in orders
                    for mode in MODES
                },
            }
        return out

    # ---- Part A examples ---------------------------------------------------------------------------
    def examples(self, examples: dict, out_dir: Path) -> None:
        for image, ex in examples.items():
            i, dims = ex["i"], ex["M"]
            empty = self.cohort.empty(i)
            k = int(np.argmax(np.where(~empty, dims.sum(1), -np.inf)))
            top_dims = np.argsort(-dims[k])[:3]
            per_dim = {}
            for d in top_dims:
                o = self.explain_crop(i, k, int(d), 1.0 if dims[k, d] >= 0 else -1.0, ex["sign"])
                per_dim[int(d)] = o
                print(
                    f"[example {image}] crop {k} {o['coords']} dim {d} IG {dims[k, d]:+.3f} | z re-encode maxabs "
                    f"{o['zerr']:.2e} | t1 compl err {o['t1']['comp_err']:.3f} t2 {o['t2']['comp_err']:.3f} | "
                    f"corr(IG t1, rollout) {np.corrcoef(o['t1']['ig'], o['rollout'])[0, 1]:+.2f}",
                    flush=True,
                )
            first = per_dim[int(top_dims[0])]
            np.savez_compressed(
                out_dir / f"example_{image[:-5]}.npz",
                img=image,
                M=dims,
                att=ex["att"],
                logit=ex["logit"],
                empty=empty,
                coords=self.cohort.coords(i),
                decisive=k,
                top_dims=top_dims,
                M_dec=dims[k],
                ig_t1=np.array([per_dim[int(d)]["t1"]["ig"] for d in top_dims]),
                ig_t2=first["t2"]["ig"],
                rollout=first["rollout"],
                **{f"crop_curve_{key}": v for key, v in ex["curves"].items()},
            )

    # ---- Part B2: tissue crops ---------------------------------------------------------------------
    def node_level(self, out_dir: Path) -> None:
        y = self.cohort.y
        candidates = [
            (i, k) for i in range(len(self.cohort.bags)) for k in np.where(~self.cohort.empty(i))[0]
        ]
        by_class = {c: [t for t in candidates if y[t[0]] == c] for c in [0, 1]}
        picks = []
        for c in [0, 1]:
            idx = self.rng.choice(
                len(by_class[c]), size=min(self.n_node_crops // 2, len(by_class[c])), replace=False
            )
            picks += [by_class[c][j] for j in idx]

        bag_ig, records = {}, []
        for q, (i, k) in enumerate(picks):
            if i not in bag_ig:
                head, bag = self.head(i), self.bag(i)
                with torch.no_grad():
                    logit = head(bag)[0][0, 0].item()
                sign = 1.0 if logit > 0 else -1.0
                attr = (
                    IntegratedGradients(lambda b, head=head, sign=sign: head(b)[0][:, 0] * sign)
                    .attribute(bag, baselines=torch.zeros_like(bag), n_steps=IG_STEPS)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                bag_ig[i] = (attr, sign, logit)
            attr, sign, logit = bag_ig[i]
            d = int(np.argmax(np.abs(attr[k])))
            o = self.explain_crop(i, k, d, 1.0 if attr[k, d] >= 0 else -1.0, sign)
            rec = {
                "img": self.cohort.names[i],
                "y": y[i],
                "crop": k,
                "dim": d,
                "logit": logit,
                "zerr": o["zerr"],
                "corr_ig_roll_t1": np.corrcoef(o["t1"]["ig"], o["rollout"])[0, 1],
                "corr_ig_roll_t2": np.corrcoef(o["t2"]["ig"], o["rollout"])[0, 1],
            }
            for t in ["t1", "t2"]:
                rec[f"{t}_comp_err"] = o[t]["comp_err"]
                rec[f"{t}_dfx"] = o[t]["fx"] - o[t]["f0"]
                for m in ["ig", "rollout"]:
                    for mode in MODES:
                        curve = o[t]["curves"][f"{m}_{mode}"]
                        rec[f"{t}_{m}_{mode}"] = auc(curve)
                        rec[f"{t}_{m}_{mode}_curve"] = curve
                for mode in MODES:
                    curves = np.array([
                        o[t]["curves"][f"random{r}_{mode}"] for r in range(N_RANDOM)
                    ])
                    rec[f"{t}_random_{mode}"] = float(np.mean([auc(c) for c in curves]))
                    rec[f"{t}_random_{mode}_curve"] = curves.mean(0)
            records.append(rec)
            if q % 20 == 0:
                print(
                    f"  B2 {q}/{len(picks)} {self.cohort.names[i]} crop {k} dim {d} zerr {o['zerr']:.1e} | "
                    + " | ".join(
                        f"{t} del ig/roll/rnd {rec[f'{t}_ig_deletion']:.2f}/{rec[f'{t}_rollout_deletion']:.2f}/"
                        f"{rec[f'{t}_random_deletion']:.2f} compl {rec[f'{t}_comp_err']:.2f}"
                        for t in ["t1", "t2"]
                    ),
                    flush=True,
                )

        scalars = {
            k: np.array([r[k] for r in records]) for k in records[0] if not k.endswith("_curve")
        }
        curves = {k: np.stack([r[k] for r in records]) for k in records[0] if k.endswith("_curve")}
        np.savez_compressed(out_dir / "faith_nodes.npz", **scalars, **curves)
        for t in ["t1", "t2"]:
            ok = np.isfinite(scalars[f"{t}_ig_deletion"]) & (np.abs(scalars[f"{t}_dfx"]) > 1e-3)
            print(
                f"B2 {t} (n={ok.sum()}): "
                + ", ".join(
                    f"{m} del {np.nanmean(scalars[f'{t}_{m}_deletion'][ok]):.3f} "
                    f"ins {np.nanmean(scalars[f'{t}_{m}_insertion'][ok]):.3f}"
                    for m in ["ig", "rollout", "random"]
                )
                + f" | median IG compl err {np.median(scalars[f'{t}_comp_err']):.3f}",
                flush=True,
            )


def main(n_node_crops: int, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    analysis = Analysis(n_node_crops)
    cohort, examples = analysis.every_bag()
    np.savez_compressed(
        out / "cohort.npz",
        img=np.array(cohort["img"]),
        y=np.array(cohort["y"]),
        logit=np.array(cohort["logit"]),
        dimtot=np.array(cohort["dimtot"]),
        dimabs=np.array(cohort["dimabs"]),
        comp_err=np.array(cohort["comp_err"]),
        b1=np.array(cohort["b1"]),
        b1_cols=np.array([f"{m}_{mode}" for m in CROP_METHODS for mode in MODES]),
        classes=np.array(analysis.cohort.classes),
    )
    b1 = np.array(cohort["b1"])
    ok = ~np.isnan(b1).any(1)
    print(
        "B1 crop-level mean AUC (del, ins):",
        {
            c: round(float(v), 3)
            for c, v in zip(
                ["ig_del", "ig_ins", "att_del", "att_ins", "rnd_del", "rnd_ins"],
                b1[ok].mean(0),
                strict=False,
            )
        },
        f"| IG completeness median rel err {np.median(cohort['comp_err']):.4f}",
        flush=True,
    )
    analysis.examples(examples, out)
    analysis.node_level(out)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--node-crops", type=int, default=200, help="tissue crops for the node-level analysis (B2)"
    )
    ap.add_argument("--out-dir", default=str(paths.CACHE / "latent_faith"))
    a = ap.parse_args()
    main(a.node_crops, a.out_dir)
