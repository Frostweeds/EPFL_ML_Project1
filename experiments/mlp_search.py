"""Regularized MLP: small CV grid (width, epochs, weight decay), then hold-out check + submission.

Usage: python experiments/mlp_search.py [dataset_dir]   (default: data/experiment_5)
Writes experiments/results/mlp_search_<dataset>.json and submission_mlp.csv.
"""

import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baselines_cv import ROOT, best_threshold, split  # noqa: E402
from capacity import cross_validate_scores, mlp  # noqa: E402
from final_eval import metrics  # noqa: E402
from helpers import create_csv_submission  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "data/experiment_5"
GRID = [  # (hidden, epochs, weight decay): small nets, few epochs, stronger decay
    (h, e, wd) for h in [16, 32, 64] for e in [1, 2, 3] for wd in [1e-4, 1e-3]
]


def main():
    tx, y = np.load(f"{DATASET}/x_train.npy"), np.load(f"{DATASET}/y_train.npy")
    y01 = (y == 1).astype(float)
    tr, te = split(len(y))

    # 1. Grid search by 3-fold CV on the 80% split
    results = {}
    for h, e, wd in GRID:
        folds = cross_validate_scores(mlp(h, e, weight_decay=wd), tx[tr], y01[tr])
        f = [r["f1"] for r in folds]
        results[f"{h},{e},{wd}"] = folds
        print(f"hidden {h:>3}, epochs {e}, wd {wd:g}: F1 {np.mean(f):.4f} ± {np.std(f):.4f} | "
              f"train {np.mean([r['f1_train'] for r in folds]):.4f}", flush=True)
    best = max(results, key=lambda key: np.mean([r["f1"] for r in results[key]]))
    h, e, wd = int(best.split(",")[0]), int(best.split(",")[1]), float(best.split(",")[2])
    model = mlp(h, e, weight_decay=wd)
    print(f"best: hidden {h}, epochs {e}, wd {wd:g}", flush=True)

    # 2. Hold-out check (train on 80%, test once on 20%)
    s_tr, s_te = model(tx[tr], y01[tr], tx[te])
    hold = metrics(y01[te], s_te >= best_threshold(s_tr, y01[tr]))
    print(f"hold-out: F1 {hold['f1']:.4f}, precision {hold['precision']:.4f}, "
          f"recall {hold['recall']:.4f}, accuracy {hold['accuracy']:.4f}", flush=True)

    # 3. Retrain on the full training set and write the submission
    tx_test = np.load(f"{DATASET}/x_test.npy")
    ids = np.genfromtxt(f"{ROOT}/data/x_test.csv", delimiter=",", skip_header=1, usecols=0, dtype=int)
    s_all, s_test = model(tx, y01, tx_test)
    pred = s_test >= best_threshold(s_all, y01)
    create_csv_submission(ids, np.where(pred, 1, -1), f"{ROOT}/submission_mlp.csv")
    print(f"written submission_mlp.csv: {pred.mean():.4f} positive")

    out = f"{ROOT}/experiments/results/mlp_search_{os.path.basename(DATASET.rstrip('/'))}.json"
    with open(out, "w") as fh:
        json.dump({"cv": results, "best": best, "holdout": hold}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
