"""Final evaluation of the selected model on the 20% hold-out split (used once).

Selected model: logistic regression, lambda = 0, gamma = 1, 300 GD steps, threshold maximizing
F1 on the training data. Trained on the 80% split, evaluated on the 20% split.

Usage: python experiments/final_eval.py [dataset_dir ...]   (default: data/experiment_5 data/experiment_6)
Writes experiments/results/final_eval.json and prints a LaTeX table.
"""

import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baselines_cv import ROOT, best_threshold, cross_validate, f1, impl, split  # noqa: E402

DATASETS = sys.argv[1:] or ["data/experiment_5", "data/experiment_6"]


def fit(tx, y, y01):
    """Selected model. In: tx (N, D), y (N,) in {-1,1}, y01 (N,) in {0,1}. Out: (w (D,), loss)."""
    return impl.logistic_regression(y01, tx, np.zeros(tx.shape[1]), 300, 1.0)


def metrics(y, pred):
    """Hold-out metrics. In: y (N,) in {0,1}, pred (N,) bool. Out: dict."""
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "f1": f1(y, pred),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "accuracy": (tp + tn) / len(y),
        "predicted_positive_rate": float(np.mean(pred)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def evaluate(dataset):
    """CV on the 80% split, then train on 80% and test once on 20%. In: dataset dir. Out: dict."""
    tx, y = np.load(f"{dataset}/x_train.npy"), np.load(f"{dataset}/y_train.npy")
    tr, te = split(len(y))
    y01 = (y == 1).astype(float)

    cv = [f["f1"] for f in cross_validate(fit, tx[tr], y[tr])]

    w, _ = fit(tx[tr], y[tr], y01[tr])  # train on the 80% split
    t = best_threshold(tx[tr] @ w, y01[tr])  # threshold on the 80% split
    holdout = metrics(y01[te], tx[te] @ w >= t)  # the 20% split, used once
    return {
        "cv_f1": np.mean(cv),
        "cv_f1_std": np.std(cv),
        "threshold_sigma": 1 / (1 + np.exp(-t)),
        "holdout": holdout,
    }


def main():
    results = {}
    for dataset in DATASETS:
        name = os.path.basename(dataset.rstrip("/"))
        results[name] = evaluate(dataset)
        r, h = results[name], results[name]["holdout"]
        print(
            f"{name}: CV F1 {r['cv_f1']:.4f} ± {r['cv_f1_std']:.4f} | hold-out F1 {h['f1']:.4f}, "
            f"precision {h['precision']:.4f}, recall {h['recall']:.4f}, accuracy {h['accuracy']:.4f}, "
            f"predicted positive {h['predicted_positive_rate']:.4f}, sigma(t) {r['threshold_sigma']:.3f}"
        )
        print(f"  confusion: {h['confusion']}", flush=True)

    print("\nLaTeX rows (model & CV F1 & hold-out F1 & precision & recall & accuracy):")
    for name, r in results.items():
        h = r["holdout"]
        print(
            f"{name} & ${r['cv_f1']:.3f} \\pm {r['cv_f1_std']:.3f}$ & {h['f1']:.3f} & "
            f"{h['precision']:.3f} & {h['recall']:.3f} & {h['accuracy']:.3f} \\\\"
        )

    path = f"{ROOT}/experiments/results/final_eval.json"
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
