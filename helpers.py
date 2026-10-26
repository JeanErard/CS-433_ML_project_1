"""Helper functions to load the data and write a submission file.

``load_csv_data`` and ``create_csv_submission`` are the helpers provided with
the project; ``load_feature_names`` and ``split_data`` were added by us.
"""

import csv
import os

import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Load the training and test sets.

    The folder must contain ``x_train.csv``, ``y_train.csv`` and ``x_test.csv``
    with their original names.

    Args:
        data_path (str): path of the data folder.
        sub_sample (bool): if True, keep only one training sample out of 50
            (useful to quickly test the code). Default is False.

    Returns:
        x_train (np.ndarray): training data, shape (N, D).
        x_test (np.ndarray): test data, shape (N_test, D).
        y_train (np.ndarray): training labels in {-1, 1}, shape (N,).
        train_ids (np.ndarray): ids of the training samples, shape (N,).
        test_ids (np.ndarray): ids of the test samples, shape (N_test,).
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def load_feature_names(data_path):
    """Read the feature names from the header of ``x_train.csv``.

    Args:
        data_path (str): path of the data folder.

    Returns:
        np.ndarray: feature names (str), shape (D,), without the 'Id' column.
    """
    header = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), dtype=str, delimiter=",", max_rows=1
    )
    return header[1:]


def split_data(x, y, ratio, seed=1):
    """Randomly split a dataset into a training part and a validation part.

    Args:
        x (np.ndarray): data, shape (N, D).
        y (np.ndarray): labels, shape (N,).
        ratio (float): fraction of the samples (rounded down) used for training.
        seed (int): random seed. Default is 1.

    Returns:
        x_tr, x_va (np.ndarray): training and validation data.
        y_tr, y_va (np.ndarray): training and validation labels.
    """
    np.random.seed(seed)
    n_train = int(np.floor(ratio * len(y)))
    indices = np.random.permutation(len(y))
    ind_tr, ind_va = indices[:n_train], indices[n_train:]
    return x[ind_tr], x[ind_va], y[ind_tr], y[ind_va]


def create_csv_submission(ids, y_pred, name):
    """Create a submission file in the format required by AIcrowd.

    The file has two columns, 'Id' and 'Prediction'.

    Args:
        ids (np.ndarray): sample ids.
        y_pred (np.ndarray): predictions, only values -1 or 1.
        name (str): path of the csv file to create.

    Raises:
        ValueError: if y_pred contains a value other than -1 or 1.
    """
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
