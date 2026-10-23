"""Compare the 6 required methods by 3-fold CV (F1, threshold tuned on the training folds).

Usage: python experiments/baselines_cv.py [dataset_dir]   (default: data/experiment_5)
Same split and folds as notebooks/train_evaluate.ipynb (seed 1), so numbers match the README.
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(ROOT)
import implementations as impl  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "data/experiment_5"
SEED, K = 1, 3


def split(n):
    """Random 80/20 split. In: n. Out: train idx (0.8n,), validation idx (0.2n,)."""
    idx = np.random.default_rng(SEED).permutation(n)
    n_val = int(n * 0.2)
    return idx[n_val:], idx[:n_val]


def make_folds(n):
    """K random folds. In: n. Out: list of K index arrays (~n/K,)."""
    return np.array_split(np.random.default_rng(SEED).permutation(n), K)


def f1(y, pred):
    """F1 of the positive class. In: y (N,) in {0,1}, pred (N,) bool. Out: scalar."""
    tp = np.sum(pred & (y == 1))
    return 2 * tp / (np.sum(pred) + np.sum(y == 1))


def best_threshold(scores, y):
    """Threshold maximizing F1 among 100 score quantiles. In: scores (N,), y (N,). Out: scalar."""
    ts = np.quantile(scores, np.linspace(0, 1, 100, endpoint=False))
    return ts[np.argmax([f1(y, scores >= t) for t in ts])]


def lambda_max(tx):
    """Largest eigenvalue of tx^T tx (power iteration). In: tx (N, D). Out: scalar."""
    v = np.random.default_rng(SEED).standard_normal(tx.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(100):
        u = tx.T @ (tx @ v)
        lam = np.linalg.norm(u)
        v = u / lam
    return lam


def baselines(gamma_gd):
    """The 6 methods, fixed hyperparameters. Out: dict name -> fit(tx, y, y01) -> (w (D,), loss)."""
    def w0(tx):
        return np.zeros(tx.shape[1])

    return {
        "least_squares": lambda tx, y, y01: impl.least_squares(y, tx),
        "ridge (lambda=1e-3)": lambda tx, y, y01: impl.ridge_regression(y, tx, 1e-3),
        "mse_gd (300 it, gamma=1/L)": lambda tx, y, y01: impl.mean_squared_error_gd(
            y, tx, w0(tx), 300, gamma_gd
        ),
        "mse_sgd (1000 steps, gamma=1e-3)": lambda tx, y, y01: impl.mean_squared_error_sgd(
            y, tx, w0(tx), 1000, 1e-3
        ),
        "mse_sgd (1 epoch, gamma=1e-4)": lambda tx, y, y01: impl.mean_squared_error_sgd(
            y, tx, w0(tx), len(y), 1e-4
        ),
        "logistic (300 it, gamma=1)": lambda tx, y, y01: impl.logistic_regression(
            y01, tx, w0(tx), 300, 1.0
        ),
        "reg_logistic (lambda=1e-4, 300 it, gamma=1)": lambda tx, y, y01: impl.reg_logistic_regression(
            y01, tx, 1e-4, w0(tx), 300, 1.0
        ),
    }


def cross_validate(fit, tx, y):
    """K-fold CV of one method. In: fit, tx (N, D), y (N,) in {-1,1}. Out: list of K metric dicts."""
    y01 = (y == 1).astype(float)
    folds = make_folds(len(y))
    results = []
    for k in range(K):
        te = folds[k]  # held-out fold
        tr = np.concatenate([folds[i] for i in range(K) if i != k])  # training folds
        np.random.seed(SEED + k)  # SGD shuffles with np.random
        start = time.time()
        w, loss = fit(tx[tr], y[tr], y01[tr])  # fit on training folds only
        seconds = time.time() - start
        s_tr, s_te = tx[tr] @ w, tx[te] @ w
        t = best_threshold(s_tr, y01[tr])  # threshold chosen on training folds only
        results.append(
            {
                "f1": f1(y01[te], s_te >= t),  # held-out F1: the number we report
                "f1_train": f1(y01[tr], s_tr >= t),
                "f1_default": f1(y01[te], s_te >= 0),  # threshold 0 = sigmoid >= 1/2
                "accuracy": np.mean(y01[te] == (s_te >= t)),
                "threshold": t,
                "loss": loss,
                "seconds": seconds,
            }
        )
    return results


def main():
    tx, y = np.load(f"{DATASET}/x_train.npy"), np.load(f"{DATASET}/y_train.npy")
    tr, _ = split(len(y))  # the 20% validation split is not touched here
    tx, y = tx[tr], y[tr]
    print(f"{DATASET}: tx {tx.shape}, P(y=1) = {np.mean(y == 1):.4f}")

    # GD step: MSE Hessian = X^T X / N, so L = lambda_max / N and GD converges iff gamma < 2/L.
    # Logistic Hessian <= X^T X / (4N), so gamma < 8/L is guaranteed.
    L = lambda_max(tx) / len(y)
    print(f"L = {L:.3f}: MSE gamma < {2 / L:.4f} (we use 1/L), logistic gamma < {8 / L:.3f}")

    out = {"dataset": DATASET, "L_mse": L, "methods": {}}
    for name, fit in baselines(1 / L).items():
        out["methods"][name] = cross_validate(fit, tx, y)
        print(f"  done: {name}", flush=True)

    print(f"\n{'method':<46}{'F1 CV':>17}{'F1 train':>10}{'F1 @0':>8}{'acc':>7}")
    for name, folds in out["methods"].items():
        def mean(key):
            return np.mean([f[key] for f in folds])

        std = np.std([f["f1"] for f in folds])
        print(
            f"{name:<46}{mean('f1'):>9.4f} ± {std:.4f}{mean('f1_train'):>10.4f}"
            f"{mean('f1_default'):>8.4f}{mean('accuracy'):>7.4f}"
        )
        print(f"{'':<46}per fold: " + " / ".join(f"{f['f1']:.4f}" for f in folds))
    folds = make_folds(len(y))
    print(f"{'all positive':<46}{np.mean([f1(y[i] == 1, np.ones(len(i), bool)) for i in folds]):>9.4f}")

    path = f"{ROOT}/experiments/results/baselines_cv_{os.path.basename(DATASET.rstrip('/'))}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
