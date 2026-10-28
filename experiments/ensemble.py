"""Ensemble: average of the standardized scores of several models.

Usage: python experiments/ensemble.py [dataset_dir] [--submit]   (default: data/experiment_6, with the HAREHAB1 leak)
1. 3-fold CV on the 80% split: every model alone and every combination, on the same folds (paired);
2. the best combination by CV is checked once on the 20% hold-out;
3. it is retrained on the full training set and written to submission_ensemble.csv.
--submit: skip 1-2 and run step 3 only with BEST (the combination found by 1, ~10 min).
Writes experiments/results/ensemble_<dataset>.json.
"""

import json
import os
import sys
from itertools import combinations

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import capacity  # noqa: E402
import kernels  # noqa: E402
from baselines_cv import K, ROOT, SEED, best_threshold, f1, impl, lambda_max, make_folds, split  # noqa: E402
from final_eval import metrics  # noqa: E402
from helpers import create_csv_submission  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DATASET = ARGS[0] if ARGS else "data/experiment_6"
BEST = ("interactions", "nystrom_rbf", "mlp")  # best by CV on experiment_6 (results/ensemble_experiment_6.log)


def nystrom_rbf(tx_tr, y_tr, tx_te):
    """Logistic on [features, 500 Nystrom RBF features], g = 1/median. Out: scores_tr, scores_te.
    Same as kernels.nystrom_logistic, but fills one preallocated matrix (fits in memory on all 328k rows)."""
    med = kernels.median_sq_dist(tx_tr[:, 1:], np.random.default_rng(SEED))
    rng = np.random.default_rng(SEED)
    phi = kernels.nystrom(kernels.rbf(1 / med), tx_tr[rng.choice(len(tx_tr), kernels.M, replace=False), 1:])
    d = tx_tr.shape[1]

    def build(x):
        """[x, phi(x)] built in chunks. In: x (n, d). Out: (n, d + M')."""
        a = np.empty((len(x), d + phi(x[:2, 1:]).shape[1]))
        a[:, :d] = x
        for s in range(0, len(x), 20000):
            a[s:s + 20000, d:] = phi(x[s:s + 20000, 1:])
        return a

    a_tr = build(tx_tr)
    mean, std = a_tr[:, d:].mean(0), a_tr[:, d:].std(0) + 1e-12  # standardize on the training rows
    a_tr[:, d:] -= mean
    a_tr[:, d:] /= std
    L = lambda_max(a_tr) / len(y_tr)
    gamma = min(1.0, 1.9 / (L / 4))  # logistic Hessian <= L/4: safe step
    w, _ = impl.logistic_regression(y_tr, a_tr, np.zeros(a_tr.shape[1]), min(1000, int(np.ceil(300 / gamma))), gamma)
    s_tr = a_tr @ w
    del a_tr
    a_te = build(tx_te)
    a_te[:, d:] -= mean
    a_te[:, d:] /= std
    return s_tr, a_te @ w


MODELS = {
    "logistic": capacity.logistic,
    "interactions": capacity.interactions(10),
    "nystrom_rbf": nystrom_rbf,
    "mlp": capacity.mlp(32, 2, weight_decay=1e-3),
}
COMBOS = [c for r in range(1, len(MODELS) + 1) for c in combinations(MODELS, r)]


def all_scores(tx_tr, y_tr, tx_te, names=MODELS):
    """Standardized scores of each model. Out: {name: (scores_tr (N,), scores_te (M,))}."""
    out = {}
    for name in names:
        s_tr, s_te = MODELS[name](tx_tr, y_tr, tx_te)
        mean, std = s_tr.mean(), s_tr.std()  # z-score with training statistics
        out[name] = ((s_tr - mean) / std, (s_te - mean) / std)
        print(f"    {name} done", flush=True)
    return out


def combine(scores, combo):
    """Average of the standardized scores of the models in combo. Out: scores_tr, scores_te."""
    return np.mean([scores[n][0] for n in combo], 0), np.mean([scores[n][1] for n in combo], 0)


def submit(tx, y01, combo):
    """Retrain combo on the full training set, write submission_ensemble.csv."""
    tx_test = np.load(f"{DATASET}/x_test.npy")
    ids = np.genfromtxt(f"{ROOT}/data/x_test.csv", delimiter=",", skip_header=1, usecols=0, dtype=int)
    s_all, s_test = combine(all_scores(tx, y01, tx_test, combo), combo)
    pred = s_test >= best_threshold(s_all, y01)
    create_csv_submission(ids, np.where(pred, 1, -1), f"{ROOT}/submission_ensemble.csv")
    print(f"written submission_ensemble.csv: {pred.mean():.4f} positive", flush=True)


def main():
    tx, y = np.load(f"{DATASET}/x_train.npy"), np.load(f"{DATASET}/y_train.npy")
    y01 = (y == 1).astype(float)
    if "--submit" in sys.argv:
        return submit(tx, y01, BEST)
    tr80, te20 = split(len(y))
    x80, y80 = tx[tr80], y01[tr80]
    print(f"{DATASET}: train split {x80.shape}", flush=True)

    # 1. CV: all models on the same folds, then every combination
    folds = make_folds(len(y80))
    cv = {c: [] for c in COMBOS}
    for k in range(K):
        te = folds[k]
        tr = np.concatenate([folds[i] for i in range(K) if i != k])
        print(f"  fold {k}", flush=True)
        scores = all_scores(x80[tr], y80[tr], x80[te])
        for c in COMBOS:
            s_tr, s_te = combine(scores, c)
            t = best_threshold(s_tr, y80[tr])  # threshold on training scores
            cv[c].append({"f1": f1(y80[te], s_te >= t), "f1_train": f1(y80[tr], s_tr >= t)})

    print(f"\n{'combination':<52}{'F1 CV':>17}{'train':>8}   per fold")
    for c in sorted(COMBOS, key=lambda c: -np.mean([r["f1"] for r in cv[c]])):
        f = [r["f1"] for r in cv[c]]
        print(f"{' + '.join(c):<52}{np.mean(f):>9.4f} ± {np.std(f):.4f}"
              f"{np.mean([r['f1_train'] for r in cv[c]]):>8.4f}   {' / '.join(f'{v:.4f}' for v in f)}")
    best = max(COMBOS, key=lambda c: np.mean([r["f1"] for r in cv[c]]))
    print(f"\nbest: {' + '.join(best)}", flush=True)

    # 2. Hold-out check of the best combination (train on 80%, test once on 20%)
    s_tr, s_te = combine(all_scores(x80, y80, tx[te20], best), best)
    hold = metrics(y01[te20], s_te >= best_threshold(s_tr, y80))
    print(f"hold-out: F1 {hold['f1']:.4f}, precision {hold['precision']:.4f}, recall {hold['recall']:.4f}, "
          f"accuracy {hold['accuracy']:.4f}", flush=True)

    out = f"{ROOT}/experiments/results/ensemble_{os.path.basename(DATASET.rstrip('/'))}.json"
    with open(out, "w") as fh:
        json.dump({"cv": {" + ".join(c): r for c, r in cv.items()}, "best": " + ".join(best),
                   "holdout": hold}, fh, indent=1, default=float)

    # 3. Retrain on the full training set and write the submission
    submit(tx, y01, best)


if __name__ == "__main__":
    main()
