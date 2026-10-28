"""Selected polynomial features and the homework logistic GD methods."""

from itertools import combinations_with_replacement

import numpy as np

from implementations import logistic_regression, reg_logistic_regression, compute_logistic_gradient
from model import find_best_threshold

WEIGHT_FRACTION = 0.5
MAX_ITERS = 5000
GAMMA = 0.2


def select_features(tx, w):
    """Magnitude of the linear coefficient for one training standard deviation."""
    strength = np.abs(w) * tx.std(axis=0)
    order = np.argsort(strength[1:])[::-1] + 1  # exclude the bias
    cumulative = np.cumsum(strength[order])
    if cumulative[-1] == 0:
        raise ValueError("All feature weights are zero.")
    count = np.searchsorted(cumulative, WEIGHT_FRACTION * cumulative[-1]) + 1
    return order[:count], strength


def polynomial_features(x_train, x_valid, degree):
    """All quadratic terms, then individual powers only. Fit scaling on training rows."""
    limits = [len(np.unique(x_train[:, j])) - 1 for j in range(x_train.shape[1])]
    lower = np.min(x_train, axis=0)
    scale = np.max(x_train, axis=0) - lower
    scale[scale == 0] = 1
    x_train, x_valid = (x_train - lower) / scale, (x_valid - lower) / scale
    terms, widths = [], []
    for power in range(1, degree + 1):
        candidates = combinations_with_replacement(range(len(scale)), power) if power <= 2 else [(j,) * power for j in range(len(scale))]
        terms.extend(term for term in candidates if all(term.count(j) <= limits[j] for j in set(term)))
        widths.append(1 + len(terms))
    train = np.ones((len(x_train), 1 + len(terms)), order="F")
    valid = np.ones((len(x_valid), 1 + len(terms)), order="F")
    for j, term in enumerate(terms, 1):
        u, v = np.prod(x_train[:, term], axis=1), np.prod(x_valid[:, term], axis=1)
        mean, std = u.mean(), u.std()
        if std < 1e-12:
            std = 1
        train[:, j], valid[:, j] = (u - mean) / std, (v - mean) / std
    return train, valid, widths


def logistic_objective(y, scores, w, lambda_):
    """Team convention: mean logistic loss + lambda * ||w||^2, including the bias."""
    return np.mean(np.logaddexp(0, scores) - y * scores) + lambda_ * (w @ w)


def fit_polynomial(y, tx, initial_w, lambda_, max_iters=MAX_ITERS, gamma=GAMMA):
    """Call the homework GD method with fixed settings and report its final gradient."""
    if lambda_ == 0:
        w, _ = logistic_regression(y, tx, initial_w, max_iters, gamma)
    else:
        w, _ = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
    gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
    gradient_norm = float(np.max(np.abs(gradient)))
    if not np.isfinite(gradient_norm):
        raise RuntimeError("Non-finite GD gradient: reduce GAMMA.")
    return w, max_iters, gradient_norm


def quadratic_features(x_train, x_valid, columns):
    """Keep all linear inputs; append scaled quadratic terms, using training statistics."""
    q_train, q_valid, widths = polynomial_features(x_train[:, columns], x_valid[:, columns], 2)
    start, p = widths[0], x_train.shape[1]
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std < 1e-12] = 1
    mean[0], std[0] = 0, 1  # preserve the original bias
    train = np.empty((len(x_train), p + q_train.shape[1] - start), order="F")
    valid = np.empty((len(x_valid), train.shape[1]), order="F")
    train[:, :p], valid[:, :p] = (x_train - mean) / std, (x_valid - mean) / std
    train[:, p:], valid[:, p:] = q_train[:, start:], q_valid[:, start:]
    return train, valid


def append_cubic(A, B, x_train, x_valid):
    """Append cubic terms for a small input subset; reuse training scaling."""
    limits = [len(np.unique(x_train[:, j])) - 1 for j in range(x_train.shape[1])]
    lower, scale = x_train.min(axis=0), np.ptp(x_train, axis=0)
    scale[scale == 0] = 1
    x_train, x_valid = (x_train - lower) / scale, (x_valid - lower) / scale
    terms = [term for term in combinations_with_replacement(range(len(scale)), 3)
             if all(term.count(j) <= limits[j] for j in set(term))]
    p = A.shape[1]
    train = np.empty((len(A), p + len(terms)), order="F")
    valid = np.empty((len(B), p + len(terms)), order="F")
    train[:, :p], valid[:, :p] = A, B
    for j, term in enumerate(terms, p):
        u, v = np.prod(x_train[:, term], axis=1), np.prod(x_valid[:, term], axis=1)
        mean, std = u.mean(), u.std()
        std = std if std >= 1e-12 else 1
        train[:, j], valid[:, j] = (u - mean) / std, (v - mean) / std
    return train, valid, terms


def fit_quadratic(y, train, p, lambda_=1e-3, max_iters=MAX_ITERS, gamma=GAMMA):
    """Fit the linear initialization and then the quadratic model with the same loss."""
    w, linear_steps, linear_gradient = fit_polynomial(y, train[:, :p], np.zeros(p), lambda_, 1000, 1.0)
    linear_objective = logistic_objective(y, train[:, :p] @ w, w, lambda_)
    w, steps, gradient = fit_polynomial(y, train, np.pad(w, (0, train.shape[1] - p)), lambda_, max_iters, gamma)
    assert logistic_objective(y, train @ w, w, lambda_) <= linear_objective + 1e-7
    info = dict(optimizer="homework_fixed_step_gd", gamma=float(gamma), max_iters=int(max_iters),
                weight_fraction=WEIGHT_FRACTION,
                lambda_=float(lambda_), original_columns=p, columns=train.shape[1],
                steps=int(steps), gradient=float(gradient),
                linear_steps=int(linear_steps), linear_gamma=1.0, linear_gradient=float(linear_gradient))
    return w, info


def train_quadratic(y, x_train, x_test, lambda_=1e-3, max_iters=MAX_ITERS, gamma=GAMMA):
    """Keep all linear features, add selected quadratic terms and fit logistic regression."""
    p = x_train.shape[1]
    selection_w, _ = logistic_regression(y, x_train, np.zeros(p), 300, 1.0)
    columns, _ = select_features(x_train, selection_w)
    train, test = quadratic_features(x_train, x_test, columns)
    w, info = fit_quadratic(y, train, p, lambda_, max_iters, gamma)
    train_scores, test_scores = train @ w, test @ w
    info.update(selected_columns=columns.tolist(), threshold=float(find_best_threshold(train_scores, y)))
    return train_scores, test_scores, info
