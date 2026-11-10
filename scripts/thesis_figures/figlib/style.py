"""Palette and Matplotlib presets shared by all thesis figures.

Colours follow the IMC markers: DNA1 blue (encoder / node stream), CD45 red (global / invariant
stream), panCK gold (decoder, tissue).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#2b2b2b"
GREY = "#8a8a8a"
BLUE, BLUE_LIGHT, BLUE_FILL = "#2f6db5", "#9fc0e8", "#e3edf9"
RED, RED_LIGHT, RED_FILL = "#c8323e", "#eea3a9", "#fbe6e7"
GOLD, GOLD_FILL = "#a87f10", "#fbf2d6"

# Display colours of individual IMC markers in image panels.
MARKER_COLOURS = {
    "DNA1": "#4C9BE8",
    "panCK": "#E8C34C",
    "CD45": "#E8584C",
    "SMA": "#5CCB7A",
    "CD8a": "#E04CC8",
    "CD4": "#4CD8E0",
    "CD3": "#F08A3C",
    "CD68": "#B07CF0",
}
SLIDE_COMPOSITE = ["DNA1", "panCK", "CD45", "SMA"]

CLASS_COLOURS = {"Adenocarcinoma": BLUE, "Squamous cell carcinoma": GOLD}
METHOD_COLOURS = {"ig": GOLD, "attention": RED, "rollout": RED, "random": GREY}

# Partition of the 43 cords markers into biological groups (marker-ablation analysis).
MARKER_GROUPS = {
    "epithelial/tumour": ["panCK", "K-Cadherin", "Carbonic Anhydrase", "Ki67"],
    "nuclear": ["DNA1", "DNA2", "Histone H3"],
    "immune": [
        "CD45",
        "CD3",
        "CD4",
        "CD8a",
        "FOXP3",
        "PD1",
        "TCF1/7",
        "CD20",
        "CD68",
        "CD15",
        "MPO",
        "HLADR",
        "Ido1",
    ],
    "stroma/fibroblast": [
        "Cadherin-11",
        "Caveolin-1",
        "CD10",
        "CD146",
        "CD248",
        "CD73",
        "Collagen Type I",
        "FAP",
        "FSP1",
        "MMP11",
        "MMP9",
        "p75",
        "PDGFRb",
        "SMA",
        "Vimentin",
        "CXCL12",
    ],
    "vascular/lymphatic": ["CD31-vWF", "CD34", "LYVE1", "PNAd", "CCL21", "VCAM1", "Podoplanin"],
}
GROUP_COLOURS = {
    "epithelial/tumour": "#c9a227",
    "nuclear": BLUE,
    "immune": RED,
    "stroma/fibroblast": "#3f9a5c",
    "vascular/lymphatic": "#e07b2c",
}
GROUP_SHORT = {g: g.split("/")[0] for g in MARKER_GROUPS} | {"epithelial/tumour": "epithelial"}

_THIN_AXES = {"axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6}
_NO_TOP_RIGHT = {"axes.spines.top": False, "axes.spines.right": False}


def use_serif(
    size: float, *, mathtext: bool = True, thin_axes: bool = False, open_axes: bool = False
) -> None:
    """Serif text (diagrams and model figures)."""
    rc = {"font.family": "serif", "font.size": size}
    if mathtext:
        rc["mathtext.fontset"] = "dejavuserif"
    if thin_axes:
        rc |= _THIN_AXES
    if open_axes:
        rc |= _NO_TOP_RIGHT
    plt.rcParams.update(rc)


def use_sans(
    size: float, *, title: float, label: float, tick: float, legend: float | None = None
) -> None:
    """Sans-serif text with open axes (explainability figures)."""
    rc = {
        "font.size": size,
        "axes.titlesize": title,
        "axes.labelsize": label,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "pdf.fonttype": 42,
        "font.family": "DejaVu Sans",
    } | _NO_TOP_RIGHT
    if legend is not None:
        rc["legend.fontsize"] = legend
    plt.rcParams.update(rc)
