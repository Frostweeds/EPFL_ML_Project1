"""Feature - feature correlation before / after the processing of experiment_13 -> correlation_heatmaps.pdf.

    python report/correlation_heatmaps.py      # from the repo root, ~2 minutes

Before: the raw columns once the special codes are NaN (normalize_codes), else 77 / 88 / 99
codes would dominate the correlations. Pairwise-complete correlation (NaN ignored per pair).
After: the 355 features of the model (bias excluded).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import build_dataset as bd  # noqa: E402
from helpers import load_csv_data, load_csv_data_labels  # noqa: E402
from normalize import normalize_codes  # noqa: E402

OUTPUT = os.path.join(os.path.dirname(__file__), "correlation_heatmaps.pdf")
MIN_ROWS = 100  # fewer rows answered by both columns -> undefined

# Diverging blue <-> red with a neutral gray midpoint
CMAP = LinearSegmentedColormap.from_list(
    "corr", ["#104281", "#3987e5", "#f0efec", "#e66767", "#9e2121"]
)
CMAP.set_bad("#ffffff")  # undefined


def pairwise_correlation(data):
    """Pearson correlation per pair of columns over the rows where both are not NaN."""
    present = ~np.isnan(data)
    centered = np.where(present, data - np.nanmean(data, axis=0), 0.0)
    m = present.astype(float)
    n = m.T @ m
    with np.errstate(divide="ignore", invalid="ignore"):
        sum_x = centered.T @ m  # [i, j]: sum of column i over rows where j is present
        sum_xx = (centered**2).T @ m
        cov = centered.T @ centered - sum_x * sum_x.T / n
        var_x = sum_xx - sum_x**2 / n
        corr = cov / np.sqrt(var_x * var_x.T)
    corr[(n < MIN_ROWS) | ~np.isfinite(corr)] = np.nan
    return np.clip(corr, -1, 1)


def draw(ax, corr, title):
    image = ax.imshow(
        corr, cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest", rasterized=True
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#52514e")
    return image


def main():
    x_labels = load_csv_data_labels(bd.DATA_DIR)
    x_train, x_test, y_train, _, _ = load_csv_data(bd.DATA_DIR)

    before = pairwise_correlation(normalize_codes(x_labels, x_train))
    settings = {**bd.DEFAULT_SETTINGS, **bd.EXPERIMENTS[13]}
    _, tx_train, _ = bd.build_features(x_labels, x_train, x_test, y_train, settings)
    after = pairwise_correlation(tx_train[:, 1:])

    for name, corr in [("before", before), ("after", after)]:
        off_diagonal = np.abs(corr[~np.eye(len(corr), dtype=bool)])
        print(
            f"{name}: {len(corr)} columns, undefined {np.isnan(off_diagonal).mean():.1%}, "
            f"|corr| >= 0.95: {np.sum(off_diagonal >= 0.95) // 2} pairs, "
            f"mean |corr| {np.nanmean(off_diagonal):.3f}"
        )

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), layout="constrained")
    draw(axes[0], before, f"(a) Raw data, {len(before)} columns")
    image = draw(axes[1], after, f"(b) After processing, {len(after)} features")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.02)
    colorbar.set_label("Pearson correlation")
    colorbar.outline.set_linewidth(0.5)
    fig.savefig(OUTPUT, dpi=300)
    print(f"{OUTPUT} written")


if __name__ == "__main__":
    main()
