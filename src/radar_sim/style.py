"""Shared matplotlib styling.

The default DejaVu Sans has no CJK glyphs, so Chinese labels would render as
boxes.  :func:`use_project_style` installs a font fallback chain and the dark
palette used by the GUI and the shipped templates.
"""

from __future__ import annotations

from matplotlib import rcParams

__all__ = ["use_project_style", "BG", "FG", "ACCENT", "DETECT", "TRUTH", "GRID"]

BG = "#0d1117"
FG = "#c9d4e2"
ACCENT = "#7bd7ff"
DETECT = "#ff5d5d"
TRUTH = "#4ade80"
GRID = "#1d2530"

_CJK_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]


def use_project_style(*, dark: bool = True) -> None:
    """Apply the CJK-capable font chain (idempotent)."""
    rcParams["font.sans-serif"] = [*_CJK_FONTS, "DejaVu Sans"]
    rcParams["font.family"] = "sans-serif"
    rcParams["axes.unicode_minus"] = False
    if dark:
        rcParams["figure.facecolor"] = BG
        rcParams["axes.facecolor"] = BG
        rcParams["savefig.facecolor"] = BG
        rcParams["axes.edgecolor"] = "#2a3441"
        rcParams["axes.labelcolor"] = FG
        rcParams["axes.titlecolor"] = FG
        rcParams["text.color"] = FG
        rcParams["xtick.color"] = FG
        rcParams["ytick.color"] = FG
        rcParams["grid.color"] = GRID
        rcParams["legend.facecolor"] = BG
        rcParams["legend.edgecolor"] = "#2a3441"
        rcParams["legend.labelcolor"] = FG
        rcParams["figure.dpi"] = 110
