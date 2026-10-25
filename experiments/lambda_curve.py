"""F1 vs regularization lambda, for ridge and regularized logistic regression (3-fold CV).

Usage: python experiments/lambda_curve.py [dataset_dir]   (default: data/experiment_5)
Same protocol as baselines_cv.py. Writes experiments/results/lambda_curve_<dataset>.json,
.pdf (for the report) and .png, and prints a LaTeX table.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baselines_cv import ROOT, cross_validate, impl, lambda_max, split  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "data/experiment_5"
LAMBDAS = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
COLORS = {"ridge": "#2a78d6", "logistic": "#eb6834"}
LABELS = {"ridge": "Ridge", "logistic": "Reg. logistic"}


def ridge(lam):
    """Ridge fit for one lambda (least squares if 0). Out: fit(tx, y, y01) -> (w (D,), loss)."""
    if lam == 0:
        return lambda tx, y, y01: impl.least_squares(y, tx)
    return lambda tx, y, y01: impl.ridge_regression(y, tx, lam)


def logistic(lam, L):
    """Reg. logistic fit by GD. In: lambda, L = lambda_max(X^T X)/N. Out: fit(tx, y, y01) -> (w (D,), loss)."""
    # lambda = 0: gamma = 1, 300 steps, as in the baselines. lambda > 0: the Hessian is
    # <= (L/4 + 2 lambda) I, so gamma < 2 / (L/4 + 2 lambda) guarantees convergence; we keep
    # gamma * max_iters ~ 300 (same "optimization time" as the baselines), at most 1000 steps.
    gamma = 1.0 if lam == 0 else min(1.0, 1.9 / (L / 4 + 2 * lam))
    max_iters = min(1000, int(np.ceil(300 / gamma)))
    return lambda tx, y, y01: impl.reg_logistic_regression(
        y01, tx, lam, np.zeros(tx.shape[1]), max_iters, gamma
    )


def summary(folds):
    """Held-out F1 (mean, std, per fold) and mean train F1. In: K fold dicts. Out: dict."""
    f1 = [f["f1"] for f in folds]
    return {
        "f1": np.mean(f1),
        "f1_std": np.std(f1),
        "f1_train": np.mean([f["f1_train"] for f in folds]),
        "f1_folds": f1,
    }


def plot(results, path):
    """F1 vs lambda: CV (solid, ± std band) and train (dashed). In: results, path without extension."""
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = [
        1e-7 if lam == 0 else lam for lam in LAMBDAS
    ]  # lambda = 0 drawn at 1e-7, labelled "0"
    for model in ["ridge", "logistic"]:
        r = results[model]
        cv = np.array([s["f1"] for s in r])
        std = np.array([s["f1_std"] for s in r])
        train = [s["f1_train"] for s in r]
        c = COLORS[model]
        ax.fill_between(x, cv - std, cv + std, color=c, alpha=0.2, linewidth=0)
        ax.plot(x, cv, color=c, lw=1.5, marker="o", ms=3, label=f"{LABELS[model]}, CV")
        ax.plot(x, train, color=c, lw=1, ls="--", label=f"{LABELS[model]}, train")
    ax.set_xscale("log")
    ax.set_xticks([1e-7, 1e-5, 1e-3, 1e-1, 10])
    ax.set_xticklabels(["0", "$10^{-5}$", "$10^{-3}$", "$10^{-1}$", "$10$"])
    ax.set_xlabel(r"regularization $\lambda$")
    ax.set_ylabel("F1")
    ax.grid(color="#e5e5e5", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=200)


def latex_table(results):
    """LaTeX table of the CV F1 per lambda. In: results. Out: str."""
    lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"$\lambda$ & Ridge & Reg.\ logistic \\",
        r"\hline",
    ]
    for i, lam in enumerate(LAMBDAS):
        cells = [
            f"${results[m][i]['f1']:.3f} \\pm {results[m][i]['f1_std']:.3f}$"
            for m in ["ridge", "logistic"]
        ]
        name = "0" if lam == 0 else f"$10^{{{int(np.log10(lam))}}}$"
        lines.append(f"{name} & {cells[0]} & {cells[1]} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    tx, y = np.load(f"{DATASET}/x_train.npy"), np.load(f"{DATASET}/y_train.npy")
    tr, _ = split(len(y))  # the 20% validation split is not touched here
    tx, y = tx[tr], y[tr]
    L = lambda_max(tx) / len(y)

    results = {"ridge": [], "logistic": []}
    for lam in LAMBDAS:
        results["ridge"].append(summary(cross_validate(ridge(lam), tx, y)))
        results["logistic"].append(summary(cross_validate(logistic(lam, L), tx, y)))
        print(
            f"lambda = {lam:g}: ridge {results['ridge'][-1]['f1']:.4f}, "
            f"logistic {results['logistic'][-1]['f1']:.4f}",
            flush=True,
        )

    name = os.path.basename(DATASET.rstrip("/"))
    path = f"{ROOT}/experiments/results/lambda_curve_{name}"
    with open(path + ".json", "w") as fh:
        json.dump({"lambdas": LAMBDAS, **results}, fh, indent=1, default=float)
    plot(results, path)
    print("\n" + latex_table(results))
    print(f"\nwritten {path}.json / .pdf / .png")


if __name__ == "__main__":
    main()
