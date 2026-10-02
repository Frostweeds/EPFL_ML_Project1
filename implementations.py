"""The 6 required ML method implementations for EPFL ML Project 1.

Every function returns (w, loss): the LAST weight vector produced by the
method (not the full history), and the corresponding loss value.
For ridge_regression and reg_logistic_regression, the returned loss must
NOT include the regularization penalty term.

All vectors are 1D numpy arrays with shape (X,), not (X, 1).
Only numpy + the Python standard library may be used (see grading tests /
project statement: no scikit-learn, no pandas, etc.).
"""

import numpy as np

from helpers import batch_iter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def compute_mse(y, tx, w):
    """Compute the mean squared error loss, with the 0.5 factor from the course notes.

    Args:
        y: shape (N,)
        tx: shape (N, D)
        w: shape (D,)

    Returns:
        loss: scalar (0-dim), 0.5 * mean((y - tx @ w) ** 2)
    """
    e = y - tx.dot(w)
    return 1 / (2 * len(y)) * e.dot(e)


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss w.r.t. w.

    Args:
        y: shape (N,)
        tx: shape (N, D)
        w: shape (D,)

    Returns:
        gradient: shape (D,)
    """
    e = y - tx.dot(w)
    return -1 / len(y) * tx.T.dot(e)


def sigmoid(t):
    """Numerically stable sigmoid, applied elementwise."""
    # TODO: implement
    raise NotImplementedError


def compute_logistic_loss(y, tx, w):
    """Compute the (unregularized) negative log-likelihood loss for logistic regression.

    Args:
        y: shape (N,), values in {0, 1}
        tx: shape (N, D)
        w: shape (D,)

    Returns:
        loss: scalar (0-dim)
    """
    # TODO: implement
    raise NotImplementedError


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic regression loss w.r.t. w."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Linear regression using gradient descent
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full-batch gradient descent.

    Args:
        y: shape (N,)
        tx: shape (N, D)
        initial_w: shape (D,), initial weight vector
        max_iters: int, number of GD steps
        gamma: float, step size

    Returns:
        w: shape (D,), last weight vector
        loss: scalar, MSE loss corresponding to w
    """
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


# ---------------------------------------------------------------------------
# 2. Linear regression using stochastic gradient descent
# ---------------------------------------------------------------------------


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (mini-batch size 1).

    Args:
        y: shape (N,)
        tx: shape (N, D)
        initial_w: shape (D,), initial weight vector
        max_iters: int, number of SGD steps
        gamma: float, step size

    Returns:
        w: shape (D,), last weight vector
        loss: scalar, MSE loss corresponding to w
    """
    w = initial_w
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_mse_gradient(batch_y, batch_tx, w)
            w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


# ---------------------------------------------------------------------------
# 3. Least squares regression using normal equations
# ---------------------------------------------------------------------------


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Allowed: anything from numpy.linalg except numpy.linalg.lstsq.

    Args:
        y: shape (N,)
        tx: shape (N, D)

    Returns:
        w: shape (D,), optimal weights
        loss: scalar, MSE loss corresponding to w
    """
    # TODO: solve tx.T @ tx @ w = tx.T @ y (e.g. via np.linalg.solve), then compute loss
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. Ridge regression using normal equations
# ---------------------------------------------------------------------------


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Args:
        y: shape (N,)
        tx: shape (N, D)
        lambda_: float, regularization parameter

    Returns:
        w: shape (D,), optimal weights
        loss: scalar, MSE loss corresponding to w (WITHOUT the penalty term)
    """
    # TODO: solve (tx.T @ tx + lambda' * I) @ w = tx.T @ y, then compute MSE loss (no penalty)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 5. Logistic regression using gradient descent
# ---------------------------------------------------------------------------


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: shape (N,), values in {0, 1}
        tx: shape (N, D)
        initial_w: shape (D,), initial weight vector
        max_iters: int, number of GD steps
        gamma: float, step size

    Returns:
        w: shape (D,), last weight vector
        loss: scalar, negative log-likelihood loss corresponding to w
    """
    w = initial_w
    # TODO: implement max_iters GD updates using compute_logistic_gradient
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 6. Regularized logistic regression using gradient descent
# ---------------------------------------------------------------------------


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent, with penalty lambda_ * ||w||^2.

    Args:
        y: shape (N,), values in {0, 1}
        tx: shape (N, D)
        lambda_: float, regularization parameter
        initial_w: shape (D,), initial weight vector
        max_iters: int, number of GD steps
        gamma: float, step size

    Returns:
        w: shape (D,), last weight vector
        loss: scalar, negative log-likelihood loss corresponding to w (WITHOUT the penalty term)
    """
    w = initial_w
    # TODO: implement max_iters GD updates on the regularized objective, e.g.
    #   grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
    #   w = w - gamma * grad
    # but report compute_logistic_loss(y, tx, w) (unregularized) as the loss.
    raise NotImplementedError
