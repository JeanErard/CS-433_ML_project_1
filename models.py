"""Our final model: class-weighted, L2-regularized logistic regression trained
with Newton's method, and the classification metrics.

Labels are in {-1, 1} here (the format of the competition), so the logistic
loss of one sample is log(1 + exp(-y * x^T w)).

The classes are very imbalanced (about 9% of positive samples). To avoid a
model that always predicts -1, each sample is weighted so that both classes
have the same total weight ("balanced" class weights).
"""

import numpy as np

from implementations import sigmoid


def balanced_class_weights(y):
    """Compute balanced sample weights: N / (2 * N_class) for each sample.

    The rare class gets a larger weight, and both classes have a total
    weight of N / 2.

    Args:
        y (np.ndarray): labels in {-1, 1}, shape (N,).

    Returns:
        np.ndarray: weight of each sample, shape (N,).
    """
    n, n_pos = len(y), np.sum(y == 1)
    return np.where(y == 1, n / (2 * n_pos), n / (2 * (n - n_pos)))


def weighted_logistic_loss(y, tx, w, weights=None):
    """Compute the mean (optionally weighted) logistic loss, y in {-1, 1}.

    Args:
        y (np.ndarray): labels in {-1, 1}, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        w (np.ndarray): weights, shape (D,).
        weights (np.ndarray or None): sample weights, shape (N,). Default is None.

    Returns:
        float: mean logistic loss.
    """
    loss = np.logaddexp(0, -y * (tx @ w))
    return np.mean(loss if weights is None else weights * loss)


def weighted_logistic_gradient(y, tx, w, lambda_=0, weights=None):
    """Compute the gradient of the weighted logistic loss + lambda_ * ||w||^2.

    Args:
        y (np.ndarray): labels in {-1, 1}, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        w (np.ndarray): weights, shape (D,).
        lambda_ (float): regularization parameter. Default is 0.
        weights (np.ndarray or None): sample weights, shape (N,). Default is None.

    Returns:
        np.ndarray: gradient, shape (D,).
    """
    n = tx.shape[0]
    weights = np.ones(n) if weights is None else weights
    r = weights * y * sigmoid(-y * (tx @ w))
    return -(tx.T @ r) / n + 2 * lambda_ * w


def weighted_logistic_hessian(tx, w, lambda_=0, weights=None):
    """Compute the Hessian of the weighted logistic loss + lambda_ * ||w||^2.

    H = 1/N * tx^T S tx + 2 * lambda_ * I, with S = diag(weights * s * (1 - s))
    and s = sigmoid(tx w).

    Args:
        tx (np.ndarray): data matrix, shape (N, D).
        w (np.ndarray): weights, shape (D,).
        lambda_ (float): regularization parameter. Default is 0.
        weights (np.ndarray or None): sample weights, shape (N,). Default is None.

    Returns:
        np.ndarray: Hessian matrix, shape (D, D).
    """
    n, d = tx.shape
    weights = np.ones(n) if weights is None else weights
    s = sigmoid(tx @ w)
    p = weights * s * (1 - s)
    return (tx.T @ (tx * p[:, None])) / n + 2 * lambda_ * np.eye(d)


def newton_logistic_regression(y, tx, initial_w, max_iters, lambda_, weights=None):
    """Train (weighted, regularized) logistic regression with Newton's method.

    Each step solves H dw = gradient and updates w <- w - dw. Newton's method
    converges in a few iterations (10-15 here) where gradient descent needs
    thousands, and it has no step size to tune.

    Args:
        y (np.ndarray): labels in {-1, 1}, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        initial_w (np.ndarray): initial weights, shape (D,).
        max_iters (int): number of Newton steps.
        lambda_ (float): regularization parameter.
        weights (np.ndarray or None): sample weights, shape (N,). Default is None.

    Returns:
        w (np.ndarray): final weights, shape (D,).
        loss (float): unweighted logistic loss on the training data.
    """
    w = initial_w.astype(float)
    for _ in range(max_iters):
        gradient = weighted_logistic_gradient(y, tx, w, lambda_, weights)
        hessian = weighted_logistic_hessian(tx, w, lambda_, weights)
        w = w - np.linalg.solve(hessian, gradient)
    return w, weighted_logistic_loss(y, tx, w)


def predict_labels(tx, w, threshold=0):
    """Predict labels in {-1, 1}: 1 if tx w >= threshold, -1 otherwise.

    Args:
        tx (np.ndarray): data matrix, shape (N, D).
        w (np.ndarray): weights, shape (D,).
        threshold (float): decision threshold on the score tx w. Default is 0.

    Returns:
        np.ndarray: predicted labels, shape (N,).
    """
    return np.where(tx @ w >= threshold, 1, -1)


def accuracy(y, y_pred):
    """Compute the fraction of correct predictions.

    Args:
        y (np.ndarray): true labels, shape (N,).
        y_pred (np.ndarray): predicted labels, shape (N,).

    Returns:
        float: accuracy in [0, 1].
    """
    return np.mean(y == y_pred)


def f1_score(y, y_pred):
    """Compute the F1 score of the positive class (label 1).

    Args:
        y (np.ndarray): true labels in {-1, 1}, shape (N,).
        y_pred (np.ndarray): predicted labels in {-1, 1}, shape (N,).

    Returns:
        float: F1 score in [0, 1] (0 if there is no true positive).
    """
    tp = np.sum((y_pred == 1) & (y == 1))
    fp = np.sum((y_pred == 1) & (y == -1))
    fn = np.sum((y_pred == -1) & (y == 1))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
