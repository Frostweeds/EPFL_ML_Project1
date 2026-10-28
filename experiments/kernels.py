"""Linear methods + kernels: exact kernel ridge regression (subsample) and Nystrom features + logistic.

Usage: python experiments/kernels.py [dataset_dir]   (default: data/experiment_5)
Protocol as in baselines_cv.py: 80/20 split (seed 1), 3-fold CV on the 80%, threshold maximizing F1
on training scores. The best Nystrom configuration is then checked once on the 20% hold-out.
Writes experiments/results/kernels_<dataset>.json.

A. Kernel ridge regression (lecture 07b): alpha = (K/n + lam I)^-1 y / n, f(x) = sum_n alpha_n k(x_n, x).
   The n x n kernel matrix is only affordable on a subsample (n = 8000 rows of each training fold).
B. Nystrom: m landmarks C, features phi(x) = k(x, C) U S^-1/2 with K_CC = U S U^T, so that
   phi(x)^T phi(x') ~ k(x, x'). Appended to the original features, then logistic regression (all rows).
"""

import json
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baselines_cv import K, ROOT, SEED, best_threshold, f1, impl, lambda_max, make_folds, split  # noqa: E402
from final_eval import metrics  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "data/experiment_5"
N_SUB = 8000  # kernel ridge: rows of each training fold
M = 500  # Nystrom: number of landmarks
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]  # kernel ridge


# ---------------------------------------------------------------------------- kernels
# x (n, D), z (m, D), standardized features without the bias column. Out: (n, m).
# Inner products and distances are divided by D so that kernel values stay O(1).


def linear(x, z):
    return x @ z.T / x.shape[1]


def poly(p):
    def k(x, z):
        return (x @ z.T / x.shape[1] + 1) ** p

    return k


def rbf(g):
    def k(x, z):
        sq = (x**2).sum(1)[:, None] + (z**2).sum(1)[None, :] - 2 * x @ z.T
        return np.exp(-g * np.maximum(sq, 0) / x.shape[1])

    return k


def kernel_chunks(kernel, x, z, alpha, chunk=5000):
    """Scores k(x, z) @ alpha without storing k(x, z). In: x (n, D), z (m, D), alpha (m,). Out: (n,)."""
    return np.concatenate([kernel(x[s:s + chunk], z) @ alpha for s in range(0, len(x), chunk)])


def median_sq_dist(x, rng, n=2000):
    """Median of ||x - x'||^2 / D over random pairs: sets the RBF scale. In: x (N, D). Out: scalar."""
    a, b = x[rng.choice(len(x), n)], x[rng.choice(len(x), n)]
    return np.median(((a - b) ** 2).sum(1) / x.shape[1])


# ---------------------------------------------------------------------------- A. kernel ridge


def krr_cv(kernel, x, y):
    """3-fold CV of kernel ridge (trained on N_SUB rows per fold) for all LAMBDAS. Out: {lam: K fold dicts}."""
    folds = make_folds(len(y))
    rng = np.random.default_rng(SEED)
    out = {lam: [] for lam in LAMBDAS}
    for k in range(K):
        te = folds[k]
        tr = rng.choice(np.concatenate([folds[i] for i in range(K) if i != k]), N_SUB, replace=False)
        y_pm = 2 * y[tr] - 1  # labels in {-1, 1}
        mean = y_pm.mean()  # intercept: fit the centered labels
        start = time.time()
        xs, xte = x[tr], x[te]
        k_tr = kernel(xs, xs)  # (n, n)
        alphas = np.column_stack([  # one column per lambda, (n, len(LAMBDAS))
            np.linalg.solve(k_tr / N_SUB + lam * np.eye(N_SUB), y_pm - mean) / N_SUB for lam in LAMBDAS
        ])
        s_tr = k_tr @ alphas
        s_te = np.concatenate([kernel(xte[s:s + 5000], xs) @ alphas for s in range(0, len(te), 5000)])
        for j, lam in enumerate(LAMBDAS):
            t = best_threshold(s_tr[:, j], y[tr])  # threshold on training scores
            out[lam].append({"f1": f1(y[te], s_te[:, j] >= t), "f1_train": f1(y[tr], s_tr[:, j] >= t)})
        print(f"    fold {k}: {time.time() - start:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------------- B. Nystrom + logistic


def nystrom(kernel, landmarks):
    """Nystrom feature map. In: kernel, landmarks (M, D). Out: phi(x) -> (n, M') with M' <= M."""
    s, u = np.linalg.eigh(kernel(landmarks, landmarks))
    keep = s > 1e-8 * s.max()  # drop numerically null directions
    proj = u[:, keep] / np.sqrt(s[keep])

    def phi(x):
        return np.concatenate([kernel(x[i:i + 5000], landmarks) @ proj for i in range(0, len(x), 5000)])

    return phi


def nystrom_logistic(kernel, tx_tr, y_tr, tx_te, rng):
    """[original features, standardized Nystrom features] + logistic regression (safe GD step).
    In: tx_tr (N, D+1) with bias column, y_tr (N,), tx_te (M, D+1). Out: scores_tr (N,), scores_te (M,)."""
    phi = nystrom(kernel, tx_tr[rng.choice(len(tx_tr), M, replace=False), 1:])
    f_tr, f_te = phi(tx_tr[:, 1:]), phi(tx_te[:, 1:])
    mean, std = f_tr.mean(0), f_tr.std(0) + 1e-12  # standardize on the training rows
    a_tr = np.hstack([tx_tr, (f_tr - mean) / std])
    a_te = np.hstack([tx_te, (f_te - mean) / std])
    del f_tr, f_te
    L = lambda_max(a_tr) / len(y_tr)
    gamma = min(1.0, 1.9 / (L / 4))  # logistic Hessian <= L/4: safe step
    w, _ = impl.logistic_regression(y_tr, a_tr, np.zeros(a_tr.shape[1]),
                                    min(1000, int(np.ceil(300 / gamma))), gamma)
    return a_tr @ w, a_te @ w


def nystrom_cv(kernel, tx, y):
    """3-fold CV of Nystrom + logistic on all training rows. Out: K fold dicts."""
    folds = make_folds(len(y))
    rng = np.random.default_rng(SEED)
    out = []
    for k in range(K):
        te = folds[k]
        tr = np.concatenate([folds[i] for i in range(K) if i != k])
        start = time.time()
        s_tr, s_te = nystrom_logistic(kernel, tx[tr], y[tr], tx[te], rng)
        t = best_threshold(s_tr, y[tr])
        out.append({"f1": f1(y[te], s_te >= t), "f1_train": f1(y[tr], s_tr >= t)})
        print(f"    fold {k}: F1 {out[-1]['f1']:.4f} ({time.time() - start:.0f}s)", flush=True)
    return out


def summary(folds):
    """Mean/std of held-out F1 and mean train F1. In: K fold dicts. Out: str."""
    f = [r["f1"] for r in folds]
    return (f"F1 {np.mean(f):.4f} ± {np.std(f):.4f} | train {np.mean([r['f1_train'] for r in folds]):.4f}"
            f" | per fold {' / '.join(f'{v:.4f}' for v in f)}")


def main():
    tx, y = np.load(f"{DATASET}/x_train.npy"), np.load(f"{DATASET}/y_train.npy")
    y01 = (y == 1).astype(float)
    tr80, te20 = split(len(y))
    x80, y80 = tx[tr80], y01[tr80]
    med = median_sq_dist(x80[:, 1:], np.random.default_rng(SEED))
    print(f"{DATASET}: train split {x80.shape}, median ||x - x'||^2 / D = {med:.3f}", flush=True)

    kernels = {
        "linear": linear,
        "poly2": poly(2),
        "poly3": poly(3),
        "rbf (g = 0.5/median)": rbf(0.5 / med),
        "rbf (g = 1/median)": rbf(1 / med),
        "rbf (g = 2/median)": rbf(2 / med),
    }
    results = {"krr": {}, "nystrom": {}}

    print(f"\nA. Kernel ridge regression, {N_SUB} training rows per fold", flush=True)
    for name, kernel in ([] if os.environ.get("SKIP_KRR") else kernels.items()):
        res = krr_cv(kernel, x80[:, 1:], y80)
        results["krr"][name] = {str(lam): r for lam, r in res.items()}
        for lam, r in res.items():
            print(f"  {name:<22} lam {lam:g}: {summary(r)}", flush=True)

    print(f"\nB. Nystrom ({M} landmarks) features + logistic regression, all training rows", flush=True)
    for name in ["poly2", "rbf (g = 0.5/median)", "rbf (g = 1/median)"]:  # the 3 best in part A
        results["nystrom"][name] = nystrom_cv(kernels[name], x80, y80)
        print(f"  {name:<22} {summary(results['nystrom'][name])}", flush=True)

    # Hold-out check of the best Nystrom kernel (train on 80%, test once on 20%)
    best = max(results["nystrom"], key=lambda n: np.mean([r["f1"] for r in results["nystrom"][n]]))
    s_tr, s_te = nystrom_logistic(kernels[best], x80, y80, tx[te20], np.random.default_rng(SEED))
    t = best_threshold(s_tr, y80)
    hold = metrics(y01[te20], s_te >= t)
    hold["f1_train"] = f1(y80, s_tr >= t)
    results["holdout"] = {"kernel": best, **hold}
    print(f"\nHold-out, Nystrom {best}: train F1 {hold['f1_train']:.4f} | hold-out F1 {hold['f1']:.4f}, "
          f"precision {hold['precision']:.4f}, recall {hold['recall']:.4f}", flush=True)

    path = f"{ROOT}/experiments/results/kernels_{os.path.basename(DATASET.rstrip('/'))}.json"
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"written {path}")


if __name__ == "__main__":
    main()
