"""Saving figures and reading the extraction caches."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save(
    fig: Figure, stem: str | Path, *, pdf_dpi: float | None = None, close: bool = True
) -> Path:
    """Write ``<stem>.pdf`` and ``<stem>.png`` (300 dpi) and return the pdf path."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    fig.savefig(pdf, **({"dpi": pdf_dpi} if pdf_dpi else {}))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    if close:
        plt.close(fig)
    print(f"wrote {pdf.with_suffix('')}.{{pdf,png}}")
    return pdf
