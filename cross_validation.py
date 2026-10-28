"""Model selection and evaluation of the report, on the features of features.py:

    1. the 6 methods of implementations.py and the quadratic model (model.py), 3-fold CV F1
    2. F1 vs the regularization lambda (ridge, regularized logistic), 3-fold CV
    3. the selected model (quadratic model) on the 20% hold-out split

Protocol: fixed 80 / 20 train / hold-out split; the 3 folds are taken in the 80% split, the
hold-out split is used once, in 3. The decision threshold is always chosen on the training
rows (F1-maximizing), then applied unchanged to the held-out rows.
Usage: python cross_validation.py [data_dir]   (default: ./data, ~40 minutes)
"""

import sys

import numpy as np

from features import build_features
from helpers import load_csv_data, load_csv_data_labels
from implementations import *
from model import f1_score, find_best_threshold, quadratic_logistic

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "./data"
SEED = 1
HOLDOUT_RATIO = 0.2
K_FOLD = 3
LAMBDAS = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]


def largest_mse_curvature(tx):
    """L = largest eigenvalue of tx^T tx / N (power iteration): GD on the MSE converges for
    gamma < 2 / L, GD on the logistic loss for gamma < 8 / L."""
    v = np.random.default_rng(SEED).standard_normal(tx.shape[1])
    v /= np.linalg.norm(v)

    for _ in range(100):
        u = tx.T @ (tx @ v)
        eigenvalue = np.linalg.norm(u)
        v = u / eigenvalue

    return eigenvalue / tx.shape[0]


def cross_validate(scores, y, y01, tx, k_indices):
    """Mean held-out F1, its std over the folds and the mean training F1.
    scores(y, tx_train, tx_test) -> (scores_train, scores_test) is trained on y: {-1, 1} for
    the MSE methods, {0, 1} (y01) for logistic; the F1 score is always computed on y01.
    """
    f1_test, f1_train = [], []

    for k in range(K_FOLD):
        test = k_indices[k]
        train = np.concatenate([k_indices[i] for i in range(K_FOLD) if i != k])

        np.random.seed(SEED + k)  # SGD shuffles with np.random
        scores_train, scores_test = scores(y[train], tx[train], tx[test])
        threshold = find_best_threshold(scores_train, y01[train])

        f1_test.append(f1_score(y01[test], (scores_test >= threshold) * 1.0))
        f1_train.append(f1_score(y01[train], (scores_train >= threshold) * 1.0))

    return np.mean(f1_test), np.std(f1_test), np.mean(f1_train)


def linear(fit):
    """fit(y, tx) -> (w, loss) as a scores function: x^T w on both sets."""

    def scores(y, tx_train, tx_test):
        w, _ = fit(y, tx_train)

        return tx_train @ w, tx_test @ w

    return scores


def format_cv(mean, std, train_f1):
    """CV F1 ± std, then training F1."""
    return f"{mean:>9.4f} ± {std:.4f}{train_f1:>10.4f}"


def ridge(lambda_):
    """Ridge regression; lambda = 0 is least squares."""
    if lambda_ == 0:
        return least_squares

    return lambda y, tx: ridge_regression(y, tx, lambda_)


def regularized_logistic(lambda_, L):
    """GD on the regularized logistic loss. lambda = 0: gamma = 1, 300 steps. lambda > 0: the
    Hessian is <= (L / 4 + 2 lambda) I, so gamma < 2 / (L / 4 + 2 lambda) converges; gamma x
    steps stays ~300 (at most 1000 steps)."""
    gamma = 1.0 if lambda_ == 0 else min(1.0, 1.9 / (L / 4 + 2 * lambda_))
    max_iters = min(1000, int(np.ceil(300 / gamma)))

    return lambda y, tx: reg_logistic_regression(
        y, tx, lambda_, np.zeros(tx.shape[1]), max_iters, gamma
    )


x_labels = load_csv_data_labels(DATA_DIR)
x_train, x_test, y_train, _ = load_csv_data(DATA_DIR)
tx, _ = build_features(x_labels, x_train, x_test, y_train)
y01 = (y_train == 1).astype(float)

permutation = np.random.default_rng(SEED).permutation(len(y_train))
n_holdout = int(len(y_train) * HOLDOUT_RATIO)
holdout, train = permutation[:n_holdout], permutation[n_holdout:]
tx_tr, y_tr, y01_tr = tx[train], y_train[train], y01[train]

k_indices = np.array_split(np.random.default_rng(SEED).permutation(len(train)), K_FOLD)

L = largest_mse_curvature(tx_tr)
zeros = np.zeros(tx.shape[1])

# 1. The 6 methods (name -> training labels, fit), then the quadratic model
methods = {
    "least_squares": (y_tr, least_squares),
    "ridge_regression (lambda = 1e-3)": (
        y_tr,
        lambda y, tx: ridge_regression(y, tx, 1e-3),
    ),
    "mean_squared_error_gd (300 steps, gamma = 1 / L)": (
        y_tr,
        lambda y, tx: mean_squared_error_gd(y, tx, zeros, 300, 1 / L),
    ),
    "mean_squared_error_sgd (10 epochs, gamma = 1e-5)": (
        y_tr,
        lambda y, tx: mean_squared_error_sgd(y, tx, zeros, 10 * len(y), 1e-5),
    ),
    "mean_squared_error_sgd (1 epoch, gamma = 1e-4)": (
        y_tr,
        lambda y, tx: mean_squared_error_sgd(y, tx, zeros, len(y), 1e-4),
    ),
    "logistic_regression (300 steps, gamma = 1)": (
        y01_tr,
        lambda y, tx: logistic_regression(y, tx, zeros, 300, 1.0),
    ),
    "reg_logistic_regression (lambda = 1e-4, 300 steps, gamma = 1)": (
        y01_tr,
        lambda y, tx: reg_logistic_regression(y, tx, 1e-4, zeros, 300, 1.0),
    ),
}

print(f"{tx_tr.shape[0]} training rows, {tx.shape[1]} features, L = {L:.3f}\n")
print(f"{'method':<64}{'CV F1':>17}{'train F1':>10}")

for name, (y_fit, fit) in methods.items():
    cv = cross_validate(linear(fit), y_fit, y01_tr, tx_tr, k_indices)
    print(f"{name:<64}{format_cv(*cv)}", flush=True)

cv = cross_validate(quadratic_logistic, y01_tr, y01_tr, tx_tr, k_indices)
print(f"{'quadratic model (model.py)':<64}{format_cv(*cv)}", flush=True)

# 2. F1 vs lambda
print(
    f"\n{'lambda':<10}{'ridge CV F1':>20}{'train F1':>10}{'reg. logistic CV F1':>24}{'train F1':>10}"
)

for lambda_ in LAMBDAS:
    ridge_cv = cross_validate(linear(ridge(lambda_)), y_tr, y01_tr, tx_tr, k_indices)
    logistic_cv = cross_validate(
        linear(regularized_logistic(lambda_, L)), y01_tr, y01_tr, tx_tr, k_indices
    )

    print(
        f"{lambda_:<10g}{format_cv(*ridge_cv):>31}{format_cv(*logistic_cv):>35}",
        flush=True,
    )

# 3. Selected model (quadratic model) on the hold-out split, used once
scores_train, scores_holdout = quadratic_logistic(y01_tr, tx_tr, tx[holdout])
threshold = find_best_threshold(scores_train, y01_tr)

prediction = scores_holdout >= threshold
truth = y01[holdout] == 1

tp, fp = np.sum(prediction & truth), np.sum(prediction & ~truth)
fn, tn = np.sum(~prediction & truth), np.sum(~prediction & ~truth)

print(
    f"\nhold-out ({len(holdout)} rows): F1 {f1_score(y01[holdout], prediction * 1.0):.4f}, "
    f"precision {tp / (tp + fp):.4f}, recall {tp / (tp + fn):.4f}, "
    f"accuracy {(tp + tn) / len(holdout):.4f} (tp {tp}, fp {fp}, fn {fn}, tn {tn})"
)
