"""Reproduce our best AIcrowd submission.

Model: class-weighted, L2-regularized logistic regression trained with Newton's
method on the preprocessed features (see preprocessing.py), with a decision
threshold tuned to maximize the F1 score.

Usage (from the root of the repository, with the data in ``data/``):
    python run.py           # use the hyperparameters found by cross-validation
    python run.py --tune    # first redo the cross-validation search (slower)

The predictions are written to ``submission.csv``.
"""

import argparse
import time

import numpy as np

from cross_validation import cross_validation_lambda_threshold
from helpers import create_csv_submission, load_csv_data, load_feature_names, split_data
from models import (
    accuracy,
    balanced_class_weights,
    f1_score,
    newton_logistic_regression,
    predict_labels,
)
from preprocessing import Preprocessor

DATA_PATH = "data"
OUTPUT_PATH = "submission.csv"

# Hyperparameter search (only used with --tune)
SPLIT_RATIO = 0.6  # fraction of the training set used for the search
SPLIT_SEED = 42
K_FOLD = 5
LAMBDAS = np.logspace(-4, -2, 5)
THRESHOLDS = np.arange(-2, 2, 0.1)
CV_ITERS = 10  # Newton steps during cross-validation

# Number of features combined two by two (interactions), see preprocessing.py
N_INTERACTIONS = 30

# Hyperparameters found by `python run.py --tune`
# (cross-validation F1 = 0.425, hold-out F1 = 0.432, accuracy = 0.878)
BEST_LAMBDA = 1e-3  # = LAMBDAS[2]
BEST_THRESHOLD = 1.1
FINAL_ITERS = 15  # Newton steps for the final model


def tune_hyperparameters(x, y, feature_names):
    """Choose lambda and the decision threshold by 5-fold cross-validation.

    For each lambda, the threshold maximizing the cross-validation F1 score is
    found; the pair (lambda, threshold) with the best F1 score is kept.
    The search is done on 60% of the training set; the other 40% is kept as a
    hold-out set to report the F1 score and accuracy of the chosen model.

    Args:
        x (np.ndarray): raw training data, shape (N, D).
        y (np.ndarray): labels in {-1, 1}, shape (N,).
        feature_names (np.ndarray): names of the D raw features.

    Returns:
        best_lambda (float): chosen regularization parameter.
        best_threshold (float): chosen decision threshold.
    """
    x_tr, x_va, y_tr, y_va = split_data(x, y, SPLIT_RATIO, seed=SPLIT_SEED)
    preprocessor = Preprocessor(n_interactions=N_INTERACTIONS)
    tx_tr = preprocessor.fit_transform(x_tr, feature_names, y_tr)
    tx_va = preprocessor.transform(x_va)
    del x_tr, x_va

    best_lambda, best_threshold, f1_cv = cross_validation_lambda_threshold(
        y_tr, tx_tr, K_FOLD, LAMBDAS, THRESHOLDS, CV_ITERS
    )
    for lambda_, f1 in zip(LAMBDAS, f1_cv.max(axis=1)):
        print(f"lambda = {lambda_:.2e}: cross-validation F1 = {f1:.4f}")
    print(
        f"Best lambda: {best_lambda!r}, best threshold: {best_threshold!r} "
        f"(cross-validation F1 = {f1_cv.max():.4f})"
    )

    w, _ = newton_logistic_regression(
        y_tr,
        tx_tr,
        np.zeros(tx_tr.shape[1]),
        FINAL_ITERS,
        best_lambda,
        balanced_class_weights(y_tr),
    )
    y_pred = predict_labels(tx_va, w, best_threshold)
    print(
        f"Hold-out set: F1 = {f1_score(y_va, y_pred):.4f}, "
        f"accuracy = {accuracy(y_va, y_pred):.4f}"
    )
    return best_lambda, best_threshold


def main():
    """Train the final model on the whole training set and write the predictions."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--tune", action="store_true", help="redo the hyperparameter search first"
    )
    args = parser.parse_args()
    start = time.time()

    print("Loading the data...")
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    feature_names = load_feature_names(DATA_PATH)

    best_lambda, best_threshold = BEST_LAMBDA, BEST_THRESHOLD
    if args.tune:
        print("Searching the hyperparameters by cross-validation...")
        best_lambda, best_threshold = tune_hyperparameters(
            x_train, y_train, feature_names
        )

    print("Training the final model on the whole training set...")
    preprocessor = Preprocessor(n_interactions=N_INTERACTIONS)
    tx_train = preprocessor.fit_transform(x_train, feature_names, y_train)
    tx_test = preprocessor.transform(x_test)
    del x_train, x_test
    print(f"Number of features after preprocessing: {tx_train.shape[1]}")

    w, _ = newton_logistic_regression(
        y_train,
        tx_train,
        np.zeros(tx_train.shape[1]),
        FINAL_ITERS,
        best_lambda,
        balanced_class_weights(y_train),
    )
    y_pred = predict_labels(tx_test, w, best_threshold)
    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)
    print(
        f"Predictions written to {OUTPUT_PATH} "
        f"({np.mean(y_pred == 1):.1%} predicted positive, {time.time() - start:.0f} s)"
    )


if __name__ == "__main__":
    main()
