"""Data preprocessing pipeline.

All the statistics (columns to drop, categories, means, standard deviations,
correlated columns) are learned on the training set only with
``Preprocessor.fit`` and then applied identically to any other set with
``Preprocessor.transform``. This guarantees that the training and test
matrices have exactly the same columns.

Pipeline, in order:
    1. drop the columns with too many missing values (NaN);
    2. drop the (almost) constant columns;
    3. one-hot encode the categorical columns (few distinct values),
       with an extra indicator column for NaN when needed;
    4. standardize the remaining (continuous) columns;
    5. replace the remaining NaN by 0 (= the mean after standardization);
    6. drop the columns that are highly correlated with an earlier column;
    7. add a column of ones (intercept).
"""

import numpy as np


class Preprocessor:
    """Learn the preprocessing on a training set and apply it to other sets.

    Args:
        nan_threshold (float): a column is dropped if its fraction of NaN is
            >= nan_threshold. Default is 0.95.
        const_threshold (float): a column is dropped if its most frequent
            (non-NaN) value represents >= const_threshold of its non-NaN
            values. Default is 0.99.
        max_categories (int): non-binary columns with at most this number of
            distinct values (NaN counted as one value) are one-hot encoded.
            Default is 5.
        corr_threshold (float): a column is dropped if its absolute
            correlation with an earlier column is >= corr_threshold.
            Default is 0.99.
    """

    def __init__(
        self,
        nan_threshold=0.95,
        const_threshold=0.99,
        max_categories=5,
        corr_threshold=0.99,
    ):
        self.nan_threshold = nan_threshold
        self.const_threshold = const_threshold
        self.max_categories = max_categories
        self.corr_threshold = corr_threshold

    def fit_transform(self, x, feature_names):
        """Learn the preprocessing on x and return the transformed x.

        Args:
            x (np.ndarray): raw training data, shape (N, D).
            feature_names (np.ndarray): names of the D raw features.

        Returns:
            np.ndarray: preprocessed training data, shape (N, D_out).
        """
        # 1-2. Columns with too many NaN, then (almost) constant columns
        self.nan_cols = find_nan_columns(x, self.nan_threshold)
        x = np.delete(x, self.nan_cols, axis=1)
        names = np.delete(feature_names, self.nan_cols)
        self.const_cols = find_constant_columns(x, self.const_threshold)
        x = np.delete(x, self.const_cols, axis=1)
        names = np.delete(names, self.const_cols)

        # 3. One-hot encoding
        self.categories = find_categorical_columns(x, self.max_categories)
        x, names, self.cont_cols = one_hot_encode(x, self.categories, names)

        # 4-5. Standardization of the continuous columns, then NaN -> 0
        x_cont = x[:, self.cont_cols]
        self.mean = np.nanmean(x_cont, axis=0)
        self.std = np.nanstd(x_cont, axis=0)
        x = self._standardize_and_fill(x)

        # 6. Highly correlated columns
        self.corr_cols = find_correlated_columns(x, self.corr_threshold)
        x = np.delete(x, self.corr_cols, axis=1)
        self.feature_names = np.insert(np.delete(names, self.corr_cols), 0, "bias")

        # 7. Intercept
        return add_intercept(x)

    def transform(self, x):
        """Apply the preprocessing learned by ``fit_transform`` to new data.

        Args:
            x (np.ndarray): raw data, shape (N, D) with the same D raw features.

        Returns:
            np.ndarray: preprocessed data, shape (N, D_out).
        """
        x = np.delete(x, self.nan_cols, axis=1)
        x = np.delete(x, self.const_cols, axis=1)
        x, _, _ = one_hot_encode(x, self.categories)
        x = self._standardize_and_fill(x)
        x = np.delete(x, self.corr_cols, axis=1)
        return add_intercept(x)

    def _standardize_and_fill(self, x):
        """Standardize the continuous columns with the training statistics and
        replace the NaN by 0."""
        x[:, self.cont_cols] = (x[:, self.cont_cols] - self.mean) / self.std
        return np.where(np.isnan(x), 0, x)


def find_nan_columns(x, threshold):
    """Find the columns whose fraction of NaN is >= threshold.

    Args:
        x (np.ndarray): data, shape (N, D).
        threshold (float): fraction of NaN in [0, 1].

    Returns:
        np.ndarray: indices of these columns.
    """
    nan_ratio = np.isnan(x).mean(axis=0)
    return np.where(nan_ratio >= threshold)[0]


def find_constant_columns(x, threshold):
    """Find the (almost) constant columns.

    A column is (almost) constant if its most frequent non-NaN value
    represents at least ``threshold`` of its non-NaN values.

    Args:
        x (np.ndarray): data, shape (N, D).
        threshold (float): fraction in [0, 1].

    Returns:
        list of int: indices of these columns.
    """
    const_cols = []
    for j in range(x.shape[1]):
        column = x[:, j]
        counts = np.unique(column[~np.isnan(column)], return_counts=True)[1]
        if len(counts) <= 1 or counts.max() / counts.sum() >= threshold:
            const_cols.append(j)
    return const_cols


def find_categorical_columns(x, max_categories):
    """Find the columns to one-hot encode and their categories.

    A column is considered categorical if it is not binary (values in {0, 1})
    and has at most ``max_categories`` distinct values, NaN counted as one.

    Args:
        x (np.ndarray): data, shape (N, D).
        max_categories (int): maximum number of distinct values.

    Returns:
        dict: for each categorical column index, a tuple
            (categories (np.ndarray, sorted, without NaN), has_nan (bool)).
    """
    categories = {}
    for j in range(x.shape[1]):
        values = np.unique(x[:, j])
        has_nan = bool(np.isnan(values).any())
        values = values[~np.isnan(values)]
        n_values = len(values) + int(has_nan)
        is_binary = set(values).issubset({0, 1})
        if n_values <= max_categories and not is_binary:
            categories[j] = (values, has_nan)
    return categories


def one_hot_encode(x, categories, feature_names=None):
    """One-hot encode the categorical columns, keep the other ones unchanged.

    Each categorical column is replaced, at the same position, by one indicator
    column per category except the last one (the reference category, to avoid
    collinearity with the intercept), plus one indicator column for NaN if the
    column had NaN in the training set. A category never seen in training
    gives all indicators equal to 0.

    Args:
        x (np.ndarray): data, shape (N, D).
        categories (dict): output of ``find_categorical_columns``.
        feature_names (np.ndarray or None): names of the D features.

    Returns:
        x_out (np.ndarray): encoded data, shape (N, D_out).
        names (list of str or None): names of the D_out columns.
        cont_cols (list of int): indices (in x_out) of the non-encoded columns.
    """
    columns, names, cont_cols = [], [], []
    for j in range(x.shape[1]):
        name = feature_names[j] if feature_names is not None else str(j)
        if j in categories:
            values, has_nan = categories[j]
            for k, value in enumerate(values[:-1]):
                columns.append((x[:, j] == value).astype(float))
                names.append(f"{name}_cat{k}")
            if has_nan:
                columns.append(np.isnan(x[:, j]).astype(float))
                names.append(f"{name}_nan")
        else:
            cont_cols.append(len(columns))
            columns.append(x[:, j])
            names.append(name)
    x_out = np.column_stack(columns)
    return x_out, (names if feature_names is not None else None), cont_cols


def find_correlated_columns(x, threshold):
    """Find the columns highly correlated with an earlier column.

    Columns are scanned from left to right; a column is dropped if its
    absolute Pearson correlation with a previous column that is kept is
    >= threshold. The data must not contain NaN.

    Args:
        x (np.ndarray): data without NaN, shape (N, D).
        threshold (float): absolute correlation threshold.

    Returns:
        np.ndarray: indices of the columns to drop.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(x.T))
    removed = np.zeros(x.shape[1], dtype=bool)
    for i in range(x.shape[1]):
        if not removed[i]:
            removed[i + 1 :] |= corr[i, i + 1 :] >= threshold
    return np.where(removed)[0]


def add_intercept(x):
    """Add a column of ones in front of the data matrix (bias term).

    Args:
        x (np.ndarray): data, shape (N, D).

    Returns:
        np.ndarray: data with intercept, shape (N, D + 1).
    """
    return np.c_[np.ones(x.shape[0]), x]
