"""Kernel experiment: logistic regression on [features, Nystrom RBF features].

The RBF kernel k(x, z) = exp(-g ||x - z||^2 / D) is approximated with M random landmarks C
(Nystrom): phi(x) = k(x, C) U S^-1/2 with k(C, C) = U S U^T, so that phi(x)^T phi(z) ~ k(x, z).
The M features phi(x), standardized, are appended to the features of features.py, then
logistic regression (implementations.py) and the F1-maximizing threshold, as in run.py.

    1. train on the 80% split of cross_validation.py, F1 on the 20% hold-out split
    2. retrain on the whole training set, write submission_kernel.csv

Usage: python run_experiment.py [data_dir]   (default: ./data, ~6 minutes, ~8 GB of memory)
"""

import sys

import numpy as np

from features import build_features
from helpers import create_csv_submission, load_csv_data, load_csv_data_labels
from implementations import logistic_regression
from model import f1_score, find_best_threshold

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "./data"
SEED = 1
HOLDOUT_RATIO = 0.2
N_LANDMARKS = 500
CHUNK = 20000  # rows per kernel block: k(x, C) is CHUNK x N_LANDMARKS


def rbf_kernel(x, z, g):
    """exp(-g ||x - z||^2 / D). x (n, D), z (m, D) -> (n, m)."""
    sq = (x**2).sum(1)[:, None] + (z**2).sum(1)[None, :] - 2 * x @ z.T

    return np.exp(-g * np.maximum(sq, 0) / x.shape[1])


def nystrom_features(tx_train, tx_test):
    """[tx, standardized Nystrom RBF features] for both sets; landmarks, kernel scale and
    standardization are fitted on tx_train. The bias column is not used by the kernel.
    g = 1 / median of ||x - x'||^2 / D over random pairs of training rows."""
    rng = np.random.default_rng(SEED)
    x = tx_train[:, 1:]
    a, b = x[rng.choice(len(x), 2000)], x[rng.choice(len(x), 2000)]
    g = 1 / np.median(((a - b) ** 2).sum(1) / x.shape[1])

    landmarks = x[rng.choice(len(x), N_LANDMARKS, replace=False)]
    s, u = np.linalg.eigh(rbf_kernel(landmarks, landmarks, g))
    keep = s > 1e-8 * s.max()  # drop numerically null directions
    projection = u[:, keep] / np.sqrt(s[keep])

    def expand(tx):
        """Filled in chunks: the full k(tx, C) would double the memory."""
        d = tx.shape[1]
        out = np.empty((len(tx), d + projection.shape[1]))
        out[:, :d] = tx

        for i in range(0, len(tx), CHUNK):
            out[i : i + CHUNK, d:] = (
                rbf_kernel(tx[i : i + CHUNK, 1:], landmarks, g) @ projection
            )

        return out

    ftx_train, ftx_test = expand(tx_train), expand(tx_test)

    d = tx_train.shape[1]
    mean, std = ftx_train[:, d:].mean(0), ftx_train[:, d:].std(0) + 1e-12
    for ftx in (ftx_train, ftx_test):
        ftx[:, d:] -= mean
        ftx[:, d:] /= std

    return ftx_train, ftx_test


def fit_logistic(tx, y01):
    """GD on the logistic loss. Its Hessian is <= L / 4 with L the largest eigenvalue of
    tx^T tx / N (power iteration), so gamma < 8 / L converges; gamma x steps stays ~300.
    Returns (w, loss)."""
    v = np.random.default_rng(SEED).standard_normal(tx.shape[1])
    v /= np.linalg.norm(v)

    for _ in range(100):
        u = tx.T @ (tx @ v)
        eigenvalue = np.linalg.norm(u)
        v = u / eigenvalue

    gamma = min(1.0, 1.9 / (eigenvalue / tx.shape[0] / 4))
    max_iters = min(1000, int(np.ceil(300 / gamma)))

    return logistic_regression(y01, tx, np.zeros(tx.shape[1]), max_iters, gamma)


def kernel_logistic(tx_train, y01, tx_test):
    """Train scores and test scores of the logistic regression on the Nystrom features."""
    ftx_train, ftx_test = nystrom_features(tx_train, tx_test)
    w, _ = fit_logistic(ftx_train, y01)

    return ftx_train @ w, ftx_test @ w


x_labels = load_csv_data_labels(DATA_DIR)
x_train, x_test, y_train, test_ids = load_csv_data(DATA_DIR)
tx_train, tx_test = build_features(x_labels, x_train, x_test, y_train)
del x_train, x_test  # raw rows no longer needed (memory)

y01 = (y_train == 1).astype(float)

# 1. Hold-out evaluation, same split as cross_validation.py
permutation = np.random.default_rng(SEED).permutation(len(y01))
n_holdout = int(len(y01) * HOLDOUT_RATIO)
holdout, train = permutation[:n_holdout], permutation[n_holdout:]

scores_train, scores_holdout = kernel_logistic(
    tx_train[train], y01[train], tx_train[holdout]
)
threshold = find_best_threshold(scores_train, y01[train])

f1_train = f1_score(y01[train], (scores_train >= threshold) * 1.0)
f1_holdout = f1_score(y01[holdout], (scores_holdout >= threshold) * 1.0)
print(f"Nystrom RBF + logistic: train F1 {f1_train:.4f}, hold-out F1 {f1_holdout:.4f}")

# 2. Whole training set -> submission
scores_train, scores_test = kernel_logistic(tx_train, y01, tx_test)
threshold = find_best_threshold(scores_train, y01)

y_pred = np.where(scores_test >= threshold, 1, -1)
create_csv_submission(test_ids, y_pred, "submission_kernel.csv")
print(f"submission_kernel.csv: {np.mean(y_pred == 1):.3f} positive")
