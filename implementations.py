"""Implementations of the six basic ML methods required for Project 1.

All functions take a label vector ``y`` of shape (N,), a data matrix ``tx`` of
shape (N, D) and return ``(w, loss)``, where ``w`` is the last weight vector of
shape (D,) and ``loss`` is the corresponding (unregularized) loss, a scalar.

Conventions (as in the lecture notes):
    * MSE loss: L(w) = 1 / (2N) * ||y - tx w||^2
    * Logistic loss (y in {0, 1}): L(w) = 1/N * sum( log(1 + exp(tx w)) - y * tx w )
    * Ridge penalty: lambda * ||w||^2, added to the loss being minimized but
      NOT included in the returned loss.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Loss functions and gradients
# ----------------------------------------------------------------------------


def compute_mse_loss(y, tx, w):
    """Compute the mean squared error loss 1/(2N) * ||y - tx w||^2.

    Args:
        y (np.ndarray): labels, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        w (np.ndarray): weights, shape (D,).

    Returns:
        float: value of the MSE loss.
    """
    error = y - tx @ w
    return error @ error / (2 * len(y))


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss with respect to w.

    Also works for a single sample (``y`` scalar and ``tx`` of shape (D,)),
    which is what stochastic gradient descent uses.

    Args:
        y (np.ndarray or float): labels, shape (N,), or a single label.
        tx (np.ndarray): data matrix, shape (N, D), or one sample of shape (D,).
        w (np.ndarray): weights, shape (D,).

    Returns:
        np.ndarray: gradient, shape (D,).
    """
    y = np.atleast_1d(y)
    tx = np.atleast_2d(tx)
    error = y - tx @ w
    return -tx.T @ error / len(y)


def sigmoid(t):
    """Compute the sigmoid function 1 / (1 + exp(-t)) element-wise.

    For very negative t, exp(-t) overflows to inf and the result is correctly
    0, so the overflow warning is silenced.

    Args:
        t (float or np.ndarray): input value(s).

    Returns:
        float or np.ndarray: values in [0, 1], same shape as t.
    """
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """Compute the mean negative log-likelihood of logistic regression.

    Uses ``np.logaddexp(0, z) = log(1 + exp(z))`` for numerical stability.

    Args:
        y (np.ndarray): labels in {0, 1}, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        w (np.ndarray): weights, shape (D,).

    Returns:
        float: value of the logistic loss.
    """
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic loss with respect to w.

    Args:
        y (np.ndarray): labels in {0, 1}, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        w (np.ndarray): weights, shape (D,).

    Returns:
        np.ndarray: gradient, shape (D,).
    """
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


# ----------------------------------------------------------------------------
# Linear regression
# ----------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full-batch gradient descent on the MSE loss.

    Args:
        y (np.ndarray): labels, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        initial_w (np.ndarray): initial weights, shape (D,).
        max_iters (int): number of gradient steps.
        gamma (float): step size.

    Returns:
        w (np.ndarray): last weight vector, shape (D,).
        loss (float): MSE loss of the last weight vector.
    """
    w = initial_w.astype(float)
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent on the MSE loss.

    Each step uses a mini-batch of size 1: one sample drawn uniformly at random.
    Use ``np.random.seed`` before calling this function for reproducible results.

    Args:
        y (np.ndarray): labels, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        initial_w (np.ndarray): initial weights, shape (D,).
        max_iters (int): number of stochastic gradient steps.
        gamma (float): step size.

    Returns:
        w (np.ndarray): last weight vector, shape (D,).
        loss (float): MSE loss of the last weight vector on the full data.
    """
    w = initial_w.astype(float)
    for _ in range(max_iters):
        n = np.random.randint(len(y))
        w = w - gamma * compute_mse_gradient(y[n], tx[n], w)
    return w, compute_mse_loss(y, tx, w)


def least_squares(y, tx):
    """Least squares regression using the normal equations (tx^T tx) w = tx^T y.

    Args:
        y (np.ndarray): labels, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).

    Returns:
        w (np.ndarray): optimal weights, shape (D,).
        loss (float): MSE loss of the optimal weights.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Minimizes 1/(2N) ||y - tx w||^2 + lambda_ * ||w||^2, whose solution is
    w = (tx^T tx + 2 N lambda_ I)^(-1) tx^T y.

    Args:
        y (np.ndarray): labels, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        lambda_ (float): regularization parameter, >= 0.

    Returns:
        w (np.ndarray): optimal weights, shape (D,).
        loss (float): MSE loss of the optimal weights (without penalty term).
    """
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T @ y)
    return w, compute_mse_loss(y, tx, w)


# ----------------------------------------------------------------------------
# Logistic regression (labels in {0, 1})
# ----------------------------------------------------------------------------


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Args:
        y (np.ndarray): labels in {0, 1}, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        initial_w (np.ndarray): initial weights, shape (D,).
        max_iters (int): number of gradient steps.
        gamma (float): step size.

    Returns:
        w (np.ndarray): last weight vector, shape (D,).
        loss (float): logistic loss of the last weight vector.
    """
    w = initial_w.astype(float)
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y in {0, 1}).

    Minimizes the logistic loss plus lambda_ * ||w||^2.

    Args:
        y (np.ndarray): labels in {0, 1}, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        lambda_ (float): regularization parameter, >= 0.
        initial_w (np.ndarray): initial weights, shape (D,).
        max_iters (int): number of gradient steps.
        gamma (float): step size.

    Returns:
        w (np.ndarray): last weight vector, shape (D,).
        loss (float): logistic loss of the last weight vector (without penalty term).
    """
    w = initial_w.astype(float)
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    return w, compute_logistic_loss(y, tx, w)
