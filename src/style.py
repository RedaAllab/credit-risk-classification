"""Shared chart styling for both notebooks: one categorical palette, one
diverging colormap, and one set of chrome defaults (grid, spines, type),
so every chart in the project reads as one system instead of a random
assortment of seaborn/matplotlib defaults.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Fixed-order categorical palette (validated: adjacent-pair CVD distance >= 8 in
# OKLab, normal-vision distance >= 15). Assign by entity identity, never by rank.
BLUE = "#2a78d6"
GREEN = "#008300"
MAGENTA = "#e87ba4"
YELLOW = "#eda100"
AQUA = "#1baf7a"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
RED = "#e34948"

CATEGORICAL = [BLUE, GREEN, MAGENTA, YELLOW, AQUA, ORANGE, VIOLET, RED]

# Binary target: non-default reads as "safe" (blue), default as "risk" (red).
TARGET_COLORS = {0: BLUE, 1: RED}
TARGET_LABELS = {0: "Non-default", 1: "Default"}

# Four-model comparison: fixed hue order, one color per model throughout.
MODEL_COLORS = {
    "Logistic (L1)": BLUE,
    "Random Forest": GREEN,
    "XGBoost": ORANGE,
    "MLP": VIOLET,
}

# Chart chrome
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"

# Sequential single-hue blue ramp, for magnitude (heatmap cells, count-based
# confusion matrices) so every "how much" chart in the project uses the same hue.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#fcfcfb", "#9ec5f4", "#3987e5", "#184f95"]
)


def sequential_from(color: str, surface: str = SURFACE) -> LinearSegmentedColormap:
    """Light-to-color sequential ramp for one entity's own hue (e.g. a model's
    confusion matrix), so magnitude encoding stays tied to that entity's identity
    color instead of an unrelated named colormap."""
    return LinearSegmentedColormap.from_list(f"sequential_{color}", [surface, color])


# One hue per model, reused for every chart that model appears in (bar, line,
# confusion matrix) so color always follows the entity, never the chart type.
MODEL_SEQUENTIAL = {name: sequential_from(color) for name, color in MODEL_COLORS.items()}

# Diverging pair for polarity (correlation heatmaps): blue <-> red through a
# neutral gray midpoint, never a hue at zero.
DIVERGING_BLUE_RED = LinearSegmentedColormap.from_list(
    "diverging_blue_red",
    ["#104281", "#6da7ec", "#f0efec", "#f2a99a", "#b3201f"],
)


def set_style():
    """Apply shared chrome defaults: hairline solid gridlines, muted axes,
    recessive spines, consistent type sizing. Call once per notebook."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS_LINE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "font.family": "sans-serif",
            "font.size": 10.5,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "figure.titlesize": 13,
            "figure.titleweight": "semibold",
        }
    )
