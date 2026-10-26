"""K-fold cross-validation to choose the hyperparameters.

* ``cross_validation_ridge``: regularization of ridge regression (baseline).
* ``cross_validation_threshold``: decision threshold on the score tx w of our
  Newton logistic regression, chosen to maximize the validation F1 score (the
  competition metric).
* ``cross_validation_lambda_threshold``: regularization and threshold of our
  model, chosen together to maximize the validation F1 score.
"""

import numpy as np

from implementations import compute_mse_loss, ridge_regression
from models import (
    balanced_class_weights,
    f1_score,
    newton_logistic_regression,
    predict_labels,
)


def build_k_indices(y, k_fold, seed):
    """Shuffle the sample indices and split them into k_fold folds.

    The N % k_fold leftover samples are not used.

    Args:
        y (np.ndarray): labels, shape (N,).
        k_fold (int): number of folds.
        seed (int): random seed.

    Returns:
        np.ndarray: indices of each fold, shape (k_fold, N // k_fold).
    """
    n = y.shape[0]
    interval = int(n / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(n)
    return np.array([indices[k * interval : (k + 1) * interval] for k in range(k_fold)])


def split_fold(k_indices, k):
    """Return the training and validation indices when fold k is held out.

    Args:
        k_indices (np.ndarray): output of ``build_k_indices``.
        k (int): index of the validation fold.

    Returns:
        ind_tr (np.ndarray): training indices.
        ind_va (np.ndarray): validation indices.
    """
    return np.delete(k_indices, k, axis=0).ravel(), k_indices[k]


def cross_validation_ridge(y, tx, k_fold, lambdas, seed=12):
    """Choose the ridge regularization parameter by k-fold cross-validation.

    Args:
        y (np.ndarray): labels, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        k_fold (int): number of folds.
        lambdas (np.ndarray): values of lambda to test.
        seed (int): random seed of the folds. Default is 12.

    Returns:
        best_lambda (float): lambda with the lowest mean validation RMSE.
        rmse_va (np.ndarray): mean validation RMSE of each lambda.
    """
    k_indices = build_k_indices(y, k_fold, seed)
    rmse_va = np.zeros(len(lambdas))
    for i, lambda_ in enumerate(lambdas):
        for k in range(k_fold):
            ind_tr, ind_va = split_fold(k_indices, k)
            w, _ = ridge_regression(y[ind_tr], tx[ind_tr], lambda_)
            rmse_va[i] += np.sqrt(2 * compute_mse_loss(y[ind_va], tx[ind_va], w))
    rmse_va /= k_fold
    return lambdas[np.argmin(rmse_va)], rmse_va


def cross_validation_threshold(
    y, tx, k_fold, lambda_, thresholds, max_iters=10, weighted=True, seed=12
):
    """Choose the decision threshold that maximizes the mean validation F1 score.

    The model is trained once per fold, then every threshold is evaluated on
    the validation fold.

    Args:
        y (np.ndarray): labels in {-1, 1}, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        k_fold (int): number of folds.
        lambda_ (float): regularization parameter.
        thresholds (np.ndarray): thresholds to test.
        max_iters (int): number of Newton steps. Default is 10.
        weighted (bool): if True, use balanced class weights. Default is True.
        seed (int): random seed of the folds. Default is 12.

    Returns:
        best_threshold (float): threshold with the highest mean F1 score.
        f1_va (np.ndarray): mean validation F1 score of each threshold.
    """
    k_indices = build_k_indices(y, k_fold, seed)
    f1_va = np.zeros(len(thresholds))
    for k in range(k_fold):
        ind_tr, ind_va = split_fold(k_indices, k)
        weights = balanced_class_weights(y[ind_tr]) if weighted else None
        w, _ = newton_logistic_regression(
            y[ind_tr], tx[ind_tr], np.zeros(tx.shape[1]), max_iters, lambda_, weights
        )
        for i, threshold in enumerate(thresholds):
            f1_va[i] += f1_score(y[ind_va], predict_labels(tx[ind_va], w, threshold))
    f1_va /= k_fold
    return thresholds[np.argmax(f1_va)], f1_va


def cross_validation_lambda_threshold(
    y, tx, k_fold, lambdas, thresholds, max_iters=10, weighted=True, seed=12
):
    """Choose the regularization and the decision threshold together.

    For each lambda, ``cross_validation_threshold`` gives the mean validation
    F1 score of every threshold; the pair (lambda, threshold) with the best
    score is returned.

    Args:
        y (np.ndarray): labels in {-1, 1}, shape (N,).
        tx (np.ndarray): data matrix, shape (N, D).
        k_fold (int): number of folds.
        lambdas (np.ndarray): values of lambda to test.
        thresholds (np.ndarray): thresholds to test.
        max_iters (int): number of Newton steps. Default is 10.
        weighted (bool): if True, use balanced class weights. Default is True.
        seed (int): random seed of the folds. Default is 12.

    Returns:
        best_lambda (float): chosen regularization parameter.
        best_threshold (float): chosen threshold.
        f1_va (np.ndarray): mean validation F1 score of each pair,
            shape (len(lambdas), len(thresholds)).
    """
    f1_va = np.zeros((len(lambdas), len(thresholds)))
    for i, lambda_ in enumerate(lambdas):
        _, f1_va[i] = cross_validation_threshold(
            y, tx, k_fold, lambda_, thresholds, max_iters, weighted, seed
        )
    i, j = np.unravel_index(np.argmax(f1_va), f1_va.shape)
    return lambdas[i], thresholds[j], f1_va
