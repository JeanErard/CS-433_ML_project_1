"""Plotting functions (matplotlib is only used for visualization)."""

import matplotlib.pyplot as plt
import numpy as np


def plot_features(x, feature_names, n_cols=6):
    """Plot the histogram of every feature, ignoring NaN values.

    Args:
        x (np.ndarray): data, shape (N, D).
        feature_names (sequence of str): names of the D features.
        n_cols (int): number of plots per row. Default is 6.
    """
    n_rows = int(np.ceil(len(feature_names) / n_cols))
    _, axs = plt.subplots(n_rows, n_cols, figsize=(20, 2.5 * n_rows), squeeze=False)
    for j, name in enumerate(feature_names):
        values = x[:, j][~np.isnan(x[:, j])]
        ax = axs[j // n_cols][j % n_cols]
        ax.hist(values, bins=max(1, int(np.sqrt(len(values)))))
        ax.set_title(f"{name} ({j})", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_cross_validation(params, scores, param_name, score_name, log_scale=True):
    """Plot a cross-validation score against the values of a hyperparameter.

    Args:
        params (np.ndarray): values of the hyperparameter.
        scores (np.ndarray): mean validation score of each value.
        param_name (str): label of the x-axis.
        score_name (str): label of the y-axis.
        log_scale (bool): if True, use a logarithmic x-axis. Default is True.
    """
    plt.figure(figsize=(5, 3.5))
    plt.plot(params, scores, marker=".")
    if log_scale:
        plt.xscale("log")
    plt.xlabel(param_name)
    plt.ylabel(score_name)
    plt.title("Cross-validation")
    plt.grid(True)
    plt.show()
