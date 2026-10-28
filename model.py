"""F1 score, F1-maximizing decision threshold and the selected (quadratic) model."""

import numpy as np

from implementations import (
    build_quadratic,
    logistic_regression,
    reg_logistic_regression,
)

N_THRESHOLDS = 100

# Seed of every random generator: hold-out split, CV folds, SGD shuffling (np.random),
# Nystrom landmarks, power iteration
SEED = 1

# Linear model: lambda = 0, gradient descent converged at 300 steps
LINEAR_MAX_ITERS = 300
LINEAR_GAMMA = 1.0

# Quadratic model: degree-2 terms of the features carrying WEIGHT_FRACTION of the linear
# weight |w| x std, then regularized logistic regression from the linear weights
WEIGHT_FRACTION = 0.5
LAMBDA = 1e-3
MAX_ITERS = 5000
GAMMA = 0.2


def f1_score(y01, prediction01):
    """F1 score of 0/1 predictions."""
    true_positive = np.sum((prediction01 == 1) & (y01 == 1))
    false_positive = np.sum((prediction01 == 1) & (y01 == 0))
    false_negative = np.sum((prediction01 == 0) & (y01 == 1))

    return 2 * true_positive / (2 * true_positive + false_positive + false_negative)


def find_best_threshold(scores, y01):
    """Threshold maximizing F1, searched among the score quantiles."""
    thresholds = np.quantile(scores, np.linspace(0, 1, N_THRESHOLDS, endpoint=False))
    f1_per_threshold = [f1_score(y01, (scores >= t).astype(float)) for t in thresholds]

    return thresholds[int(np.argmax(f1_per_threshold))]


def select_strongest(w, tx):
    """Columns carrying WEIGHT_FRACTION of the total |w| x std, strongest first (no bias)."""
    strength = np.abs(w) * tx.std(axis=0)
    strength[0] = 0  # bias
    order = np.argsort(strength)[::-1]
    cumulative = np.cumsum(strength[order])

    return order[: np.searchsorted(cumulative, WEIGHT_FRACTION * cumulative[-1]) + 1]


def quadratic_logistic(y01, tx_train, tx_test):
    """Train and test scores of the quadratic model. Selection, scaling and weights are
    fitted on tx_train only."""
    initial_w = np.zeros(tx_train.shape[1])
    w, _ = logistic_regression(y01, tx_train, initial_w, LINEAR_MAX_ITERS, LINEAR_GAMMA)

    selected = select_strongest(w, tx_train)
    q_train, q_test = build_quadratic(tx_train[:, selected], tx_test[:, selected])
    tx_train, tx_test = np.c_[tx_train, q_train], np.c_[tx_test, q_test]

    initial_w = np.r_[w, np.zeros(q_train.shape[1])]
    w, _ = reg_logistic_regression(y01, tx_train, LAMBDA, initial_w, MAX_ITERS, GAMMA)

    return tx_train @ w, tx_test @ w
