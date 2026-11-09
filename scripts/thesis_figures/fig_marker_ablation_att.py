"""Variant of fig_marker_ablation.py with an extra ABMIL-attention map next to the IG decision map.

Marker ablation figure, Sec. 6.3 deep-dive: which of the 43 IMC markers drive the decision.

Reads
  figures/_cache/spatial_v3/<tag>_<img>.npz   per-crop ABMIL IG (decision map), decisive crop,
                                                      Butterworth-filtered slide
  figures/_cache/marker_ablation/*.npz        marker / group ablation through the full chain
                                                      (raw -> ImmuVis -> PCA -> PIGVAE -> z_global -> ABMIL)
Effects are oriented to the predicted class: > 0 means removing the marker weakens the call.

Panels (one patient):
  (a) slide composite with the decisive real-tissue crop boxed
  (b) per-crop ABMIL IG decision map (empty crops hatched)
  (c) the top markers inside the decisive crop (crop-level ablation effect in the title)
  (d) slide-level marker effects for this patient (top + bottom), coloured by marker group
  (e) cohort: group effects, adenocarcinoma vs squamous, one dot per slide
Usage: fig_marker_ablation.py [adeno_7064|squam_6577]   (default adeno_7064)
"""
import rootutils
import glob, sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
CACHE = ROOT / "figures" / "_cache"
OUT = ROOT / "figures"
TAG = sys.argv[1] if len(sys.argv) > 1 else "adeno_7064"
P = 128

GROUP_COL = {"epithelial/tumour": "#c9a227", "nuclear": "#2f6db5", "immune": "#c8323e",
             "stroma/fibroblast": "#3f9a5c", "vascular/lymphatic": "#e07b2c"}
GROUP_SHORT = {"epithelial/tumour": "epithelial", "nuclear": "nuclear", "immune": "immune",
               "stroma/fibroblast": "stroma", "vascular/lymphatic": "vascular"}
CLASS_COL = {"Adenocarcinoma": "#2f6db5", "Squamous cell carcinoma": "#a87f10"}
COMPOSITE = [("DNA1", "#4C9BE8"), ("panCK", "#E8C34C"), ("CD45", "#E8584C"), ("SMA", "#5CCB7A")]
N_TILES, N_TOP, N_BOT = 5, 10, 4
GREY_TXT = "#555555"

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})

sp = np.load(CACHE / "spatial_v3" / f"{TAG}.npz")
ab = np.load(CACHE / "marker_ablation" / f"{TAG.split('_')[1]}.npz")
assert (sp["coords"] == ab["coords"]).all()
markers = list(ab["markers"]); gnames = list(ab["group_names"])
GROUPS = {  # same partition as extract_marker_ablation.py
    "epithelial/tumour": ["panCK", "K-Cadherin", "Carbonic Anhydrase", "Ki67"],
    "nuclear": ["DNA1", "DNA2", "Histone H3"],
    "immune": ["CD45", "CD3", "CD4", "CD8a", "FOXP3", "PD1", "TCF1/7", "CD20", "CD68", "CD15", "MPO", "HLADR", "Ido1"],
    "stroma/fibroblast": ["Cadherin-11", "Caveolin-1", "CD10", "CD146", "CD248", "CD73", "Collagen Type I", "FAP", "FSP1",
                          "MMP11", "MMP9", "p75", "PDGFRb", "SMA", "Vimentin", "CXCL12"],
    "vascular/lymphatic": ["CD31-vWF", "CD34", "LYVE1", "PNAd", "CCL21", "VCAM1", "Podoplanin"],
}
group_of = {m: g for g, ms in GROUPS.items() for m in ms}
slide = sp["slide"].astype(np.float32)
coords, empty, ig = sp["coords"], sp["empty"], sp["ig"]
k = int(sp["decisive"]); r0, c0 = coords[k]
pred = "squamous" if sp["logit"] > 0 else "adenocarcinoma"
label = str(ab["label"]); IMG = str(ab["img"])[:-5]


def stretch(ch, lo=1, hi=99.5):
    a, b = np.percentile(ch, lo), np.percentile(ch, hi)
    return np.clip((ch - a) / (b - a), 0, 1) if b > a else np.zeros_like(ch)


def tint(ch, hexcol, ref=None):
    rgb = np.array(matplotlib.colors.to_rgb(hexcol))
    if ref is None:
        s = stretch(ch)
    else:  # stretch with the slide-level percentiles, so tile brightness is comparable to the slide
        a, b = np.percentile(ref, 1), np.percentile(ref, 99.5)
        s = np.clip((ch - a) / (b - a), 0, 1) if b > a else np.zeros_like(ch)
    return s[..., None] * rgb


fig = plt.figure(figsize=(6.3, 7.7))
gs = fig.add_gridspec(3, 1, height_ratios=[2.35, 1.25, 2.6], hspace=0.55)
top = gs[0].subgridspec(1, 3, wspace=0.12)
mid = gs[1].subgridspec(1, N_TILES, wspace=0.06)
bot = gs[2].subgridspec(1, 2, wspace=0.45, width_ratios=[1.0, 1.15])

# (a) composite with decisive crop
ax = fig.add_subplot(top[0])
comp = np.clip(0.55 * sum(tint(slide[markers.index(m)], c) for m, c in COMPOSITE), 0, 1)
ax.imshow(comp)
ax.add_patch(Rectangle((c0 - .5, r0 - .5), P, P, fill=False, ec="white", lw=1.6))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"(a) Slide {IMG}", loc="left")
ax.text(0.5, -0.03, f"{label.split()[0].lower()}, logit {sp['logit']:+.1f}", transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
for i, (m, c) in enumerate(COMPOSITE):
    ax.text(0.01, 0.99 - 0.055 * i, m, color=c, transform=ax.transAxes, fontsize=6.5, va="top",
            fontweight="bold")
ax.plot([slide.shape[2] - 120, slide.shape[2] - 20], [slide.shape[1] - 25] * 2, color="white", lw=2)
ax.text(slide.shape[2] - 70, slide.shape[1] - 35, "100 µm", color="white", ha="center", fontsize=6)

# (b) ABMIL IG decision map
ax = fig.add_subplot(top[1])
grid = np.full(slide.shape[1:], np.nan)
real = ~empty
vmax = np.abs(ig[real]).max()
for j, (a, b) in enumerate(coords):
    grid[a:a + P, b:b + P] = ig[j] if real[j] else np.nan
im = ax.imshow(grid, cmap="cividis", vmin=0, vmax=vmax)
for j, (a, b) in enumerate(coords):
    if empty[j]:
        ax.add_patch(Rectangle((b - .5, a - .5), P, P, fill=False, hatch="////", ec="#bbbbbb", lw=0))
ax.add_patch(Rectangle((c0 - .5, r0 - .5), P, P, fill=False, ec="#c8323e", lw=1.6))
ax.set_xlim(-.5, slide.shape[2] - .5); ax.set_ylim(slide.shape[1] - .5, -.5)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(b) ABMIL IG per crop", loc="left")
for s_ in ax.spines.values(): s_.set_visible(False)
cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
cb.set_label("IG toward prediction", fontsize=6.5, labelpad=1); cb.ax.tick_params(labelsize=5.5)

# (c) ABMIL attention map (head-averaged; all crops, empty ones hatched but coloured)
att = sp["att"]
ax = fig.add_subplot(top[2])
grid = np.full(slide.shape[1:], np.nan)
for j, (a, b) in enumerate(coords):
    grid[a:a + P, b:b + P] = att[j]
im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=att.max())
for j, (a, b) in enumerate(coords):
    if empty[j]:
        ax.add_patch(Rectangle((b - .5, a - .5), P, P, fill=False, hatch="////", ec="#555555", lw=0))
ax.add_patch(Rectangle((c0 - .5, r0 - .5), P, P, fill=False, ec="k", lw=1.4, ls="--"))
ja = int(np.argmax(att)); aa, ab_ = coords[ja]
ax.add_patch(Rectangle((ab_ - .5, aa - .5), P, P, fill=False, ec="#2f6db5", lw=1.6))
rank = int((att > att[k]).sum()) + 1
ax.set_xlim(-.5, slide.shape[2] - .5); ax.set_ylim(slide.shape[1] - .5, -.5)
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_title("(c) ABMIL attention per crop", loc="left")
cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
cb.set_label(f"attention (sums to 1)", fontsize=6.5, labelpad=1)
print(f"attention: decisive crop rank {rank}, max crop {ja} empty={bool(empty[ja])}")
cb.ax.tick_params(labelsize=5.5)

# (c) top markers inside the decisive crop
ca = ab["crop_abl"][k]
order = np.argsort(-ca)[:N_TILES]
for t, mi in enumerate(order):
    ax = fig.add_subplot(mid[t])
    m = markers[mi]; col = GROUP_COL[group_of[m]]
    ax.imshow(tint(slide[mi, r0:r0 + P, c0:c0 + P], col, ref=slide[mi]))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color(col); s.set_linewidth(1.2)
    ax.set_title(f"{m.replace('Carbonic Anhydrase', 'Carb. anhydrase')}\n{ca[mi]:+.2f}", fontsize=7, color=col)
    if t == 0:
        ax.text(0, 1.42, "(d) Markers whose removal (in this crop only) weakens the call most",
                transform=ax.transAxes, fontsize=8, va="bottom")

# (d) slide-level marker effects for this patient
ax = fig.add_subplot(bot[0])
sa = ab["slide_abl"]
o = np.argsort(-sa); sel = np.r_[o[:N_TOP], o[-N_BOT:]]
yy = np.r_[np.arange(N_TOP), np.arange(N_TOP, N_TOP + N_BOT) + 0.8]
ax.barh(yy, sa[sel], color=[GROUP_COL[group_of[markers[j]]] for j in sel], height=0.75)
ax.set_yticks(yy); ax.set_yticklabels([markers[j].replace("Carbonic Anhydrase", "Carb. anhydrase") for j in sel])
ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("Δ logit, marker removed from all crops")
ax.set_title(f"(e) Marker effects, slide {IMG}", loc="left", pad=22)
# direction labels above the bars, split at x = 0
other = "adeno" if pred == "squamous" else "squamous"
pred_s = "adeno" if pred != "squamous" else "squamous"
xmin, xmax = ax.get_xlim()
trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.annotate("", xy=(xmin + 0.02 * (xmax - xmin), 1.035), xytext=(-0.05 * (xmax - xmin), 1.035), xycoords=trans,
            arrowprops=dict(arrowstyle="->", color=GREY_TXT, lw=0.8))
ax.annotate("", xy=(xmax - 0.02 * (xmax - xmin), 1.035), xytext=(0.05 * (xmax - xmin), 1.035), xycoords=trans,
            arrowprops=dict(arrowstyle="->", color=GREY_TXT, lw=0.8))
ax.text(-0.06 * (xmax - xmin), 1.06, f"marker supports {other}", transform=trans, ha="right", va="bottom",
        fontsize=6, color=GREY_TXT)
ax.text(0.06 * (xmax - xmin), 1.06, f"marker supports {pred_s}", transform=trans, ha="left", va="bottom",
        fontsize=6, color=GREY_TXT)
ax.set_xlim(xmin, xmax)
# group colour legend
used = [g for g in GROUP_COL if any(group_of[markers[j]] == g for j in sel)]
ax.legend(handles=[Rectangle((0, 0), 1, 1, color=GROUP_COL[g], label=GROUP_SHORT[g]) for g in GROUP_COL],
          title="marker group", title_fontsize=6.5, loc="lower right", frameon=False,
          handlelength=1.0, handleheight=0.8, labelspacing=0.3, borderaxespad=0.2)

# (e) cohort group effects
ax = fig.add_subplot(bot[1])
cohort = [np.load(f) for f in sorted(glob.glob(str(CACHE / "marker_ablation" / "*.npz")))]
rng = np.random.default_rng(0)
cls_names = list(CLASS_COL)
for gi, g in enumerate(gnames):
    for ci, cn in enumerate(cls_names):
        v = np.array([d["group_abl"][gi] for d in cohort if str(d["label"]) == cn])
        x = gi + (ci - 0.5) * 0.36
        ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=7, color=CLASS_COL[cn], alpha=0.75, lw=0)
        ax.hlines(v.mean(), x - 0.13, x + 0.13, color="k", lw=1.2)
    ax.axvspan(gi - 0.5, gi + 0.5, color=GROUP_COL[g], alpha=0.07, lw=0)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(len(gnames))); ax.set_xticklabels([GROUP_SHORT[g] for g in gnames], rotation=30, ha="right")
for t, g in zip(ax.get_xticklabels(), gnames): t.set_color(GROUP_COL[g])
ax.set_xlim(-0.5, len(gnames) - 0.5)
ax.set_ylabel("Δ logit toward prediction\n(marker group removed)")
nper = {cn: sum(str(d["label"]) == cn for d in cohort) for cn in cls_names}
ax.set_title("(f) Cohort, correctly classified slides", loc="left", pad=16)
ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=4, color=CLASS_COL[cn],
                              label=f"{cn.split()[0].lower()} (n={nper[cn]})") for cn in cls_names],
          loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.8)

fig.subplots_adjust(left=0.12, right=0.93, top=0.95, bottom=0.08)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_marker_ablation_att_{TAG}.{ext}", dpi=300)
print("wrote", OUT / f"fig_marker_ablation_att_{TAG}.pdf", "| decisive crop", k, (r0, c0),
      "| top in crop", [(markers[j], round(ca[j], 2)) for j in order])
