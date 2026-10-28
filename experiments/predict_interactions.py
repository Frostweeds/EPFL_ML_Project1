"""Best model (logistic + degree-2 interactions of the top-10 features): hold-out check + submission.

Usage: python experiments/predict_interactions.py [dataset_dir]   (default: data/experiment_5)
1. trains on the 80% split, reports F1 / precision / recall on the 20% hold-out (used once);
2. retrains on the full training set and writes submission_interactions.csv for AIcrowd.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baselines_cv import ROOT, best_threshold, split  # noqa: E402
from capacity import interactions  # noqa: E402
from final_eval import metrics  # noqa: E402
from helpers import create_csv_submission  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "data/experiment_5"
MODEL = interactions(10)


def fit_predict(tx_tr, y01_tr, tx_te):
    """Fit, threshold on training scores, predict. In: tx_tr (N, D), y01_tr (N,), tx_te (M, D). Out: (M,) bool."""
    s_tr, s_te = MODEL(tx_tr, y01_tr, tx_te)
    t = best_threshold(s_tr, y01_tr)  # threshold on the training data only
    print(f"  threshold sigma(t) = {1 / (1 + np.exp(-t)):.3f}", flush=True)
    return s_te >= t


def main():
    tx, y = np.load(f"{DATASET}/x_train.npy"), np.load(f"{DATASET}/y_train.npy")
    y01 = (y == 1).astype(float)

    tr, te = split(len(y))
    h = metrics(y01[te], fit_predict(tx[tr], y01[tr], tx[te]))  # the 20% hold-out, used once
    print(f"hold-out: F1 {h['f1']:.4f}, precision {h['precision']:.4f}, recall {h['recall']:.4f}, "
          f"accuracy {h['accuracy']:.4f}", flush=True)

    tx_test = np.load(f"{DATASET}/x_test.npy")
    ids = np.genfromtxt(f"{ROOT}/data/x_test.csv", delimiter=",", skip_header=1, usecols=0, dtype=int)
    pred = fit_predict(tx, y01, tx_test)  # retrain on the full training set
    path = f"{ROOT}/submission_interactions.csv"
    create_csv_submission(ids, np.where(pred, 1, -1), path)
    print(f"written {path}: {len(pred)} predictions, {pred.mean():.4f} positive")


if __name__ == "__main__":
    main()
