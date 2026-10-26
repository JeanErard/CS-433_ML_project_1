"""Data preprocessing pipeline.

All the statistics (columns to drop, categories, means, standard deviations,
correlated columns, interactions) are learned on the training set only with
``Preprocessor.fit_transform`` and then applied identically to any other set
with ``Preprocessor.transform``. This guarantees that the training and test
matrices have exactly the same columns.

Pipeline, in order:
    1. drop the survey metadata (interview date, ids, sampling weights);
    2. decode the special BRFSS answer codes ("don't know", "refused",
       "none", answers given per week or per month);
    3. drop the columns with too many missing values (NaN);
    4. drop the (almost) constant columns;
    5. one-hot encode the categorical columns (few distinct values),
       with an extra indicator column for NaN when needed;
    6. standardize the remaining (continuous) columns;
    7. replace the remaining NaN by 0 (= the mean after standardization);
    8. drop the columns that are highly correlated with an earlier column;
    9. add the pairwise products (interactions) of the columns that are the
       most correlated with the label;
    10. add a column of ones (intercept).
"""

import numpy as np

# Survey design variables: they describe how and when the interview was made
# (date, sample ids, sampling weights), not the person.
SURVEY_METADATA = [
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "QSTVER",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_DUALCOR",
    "_LLCPWT",
]


class Preprocessor:
    """Learn the preprocessing on a training set and apply it to other sets.

    The default values are the ones of our final model.

    Args:
        drop_metadata (bool): if True, drop the survey metadata columns
            listed in ``SURVEY_METADATA``. Default is True.
        decode_codes (bool): if True, decode the special BRFSS codes (see
            ``find_code_rules``). Default is True.
        nan_threshold (float): a column is dropped if its fraction of NaN is
            >= nan_threshold. Default is 0.95.
        const_threshold (float): a column is dropped if its most frequent
            (non-NaN) value represents >= const_threshold of its non-NaN
            values. Default is 0.99.
        max_categories (int): non-binary columns with at most this number of
            distinct values (NaN counted as one value) are one-hot encoded.
            Default is 10.
        corr_threshold (float): a column is dropped if its absolute
            correlation with an earlier column is >= corr_threshold.
            Default is 0.99.
        n_interactions (int): the products of all pairs among the
            n_interactions columns most correlated with the label are added
            (0 for none). Default is 30.
    """

    def __init__(
        self,
        drop_metadata=True,
        decode_codes=True,
        nan_threshold=0.95,
        const_threshold=0.99,
        max_categories=10,
        corr_threshold=0.99,
        n_interactions=30,
    ):
        self.drop_metadata = drop_metadata
        self.decode_codes = decode_codes
        self.nan_threshold = nan_threshold
        self.const_threshold = const_threshold
        self.max_categories = max_categories
        self.corr_threshold = corr_threshold
        self.n_interactions = n_interactions

    def fit_transform(self, x, feature_names, y=None):
        """Learn the preprocessing on x and return the transformed x.

        Args:
            x (np.ndarray): raw training data, shape (N, D).
            feature_names (np.ndarray): names of the D raw features.
            y (np.ndarray or None): training labels, shape (N,). Only needed
                to choose the interactions (n_interactions > 0).

        Returns:
            np.ndarray: preprocessed training data, shape (N, D_out).
        """
        # 1-2. Survey metadata and special answer codes
        self.metadata_cols = (
            np.where(np.isin(feature_names, SURVEY_METADATA))[0]
            if self.drop_metadata
            else np.array([], dtype=int)
        )
        x = np.delete(x, self.metadata_cols, axis=1)
        names = np.delete(feature_names, self.metadata_cols)
        self.code_rules = find_code_rules(x) if self.decode_codes else {}
        x = decode_special_codes(x, self.code_rules)

        # 3-4. Columns with too many NaN, then (almost) constant columns
        self.nan_cols = find_nan_columns(x, self.nan_threshold)
        x = np.delete(x, self.nan_cols, axis=1)
        names = np.delete(names, self.nan_cols)
        self.const_cols = find_constant_columns(x, self.const_threshold)
        x = np.delete(x, self.const_cols, axis=1)
        names = np.delete(names, self.const_cols)

        # 5. One-hot encoding
        self.categories = find_categorical_columns(x, self.max_categories)
        x, names, self.cont_cols = one_hot_encode(x, self.categories, names)

        # 6-7. Standardization of the continuous columns, then NaN -> 0
        x_cont = x[:, self.cont_cols]
        self.mean = np.nanmean(x_cont, axis=0)
        self.std = np.nanstd(x_cont, axis=0)
        x = self._standardize_and_fill(x)

        # 8. Highly correlated columns
        self.corr_cols = find_correlated_columns(x, self.corr_threshold)
        x = np.delete(x, self.corr_cols, axis=1)
        names = np.delete(names, self.corr_cols)

        # 9. Interactions between the columns most correlated with the label
        self.pairs = (
            find_interaction_pairs(x, y, self.n_interactions)
            if self.n_interactions > 0
            else []
        )
        # the products are standardized; constant products (e.g. two
        # categories of the same question, always 0) are useless and removed
        stats = [
            (np.mean(x[:, a] * x[:, b]), np.std(x[:, a] * x[:, b]))
            for a, b in self.pairs
        ]
        keep = [k for k, (_, std) in enumerate(stats) if std > 1e-8]
        self.pairs = [self.pairs[k] for k in keep]
        self.inter_mean = np.array([stats[k][0] for k in keep])
        self.inter_std = np.array([stats[k][1] for k in keep])
        names = list(names) + [f"{names[a]}*{names[b]}" for a, b in self.pairs]
        self.feature_names = np.array(["bias"] + names)

        # 10. Interactions and intercept
        return self._add_interactions_and_intercept(x)

    def transform(self, x):
        """Apply the preprocessing learned by ``fit_transform`` to new data.

        Args:
            x (np.ndarray): raw data, shape (N, D) with the same D raw features.

        Returns:
            np.ndarray: preprocessed data, shape (N, D_out).
        """
        x = np.delete(x, self.metadata_cols, axis=1)
        x = decode_special_codes(x, self.code_rules)
        x = np.delete(x, self.nan_cols, axis=1)
        x = np.delete(x, self.const_cols, axis=1)
        x, _, _ = one_hot_encode(x, self.categories)
        x = self._standardize_and_fill(x)
        x = np.delete(x, self.corr_cols, axis=1)
        return self._add_interactions_and_intercept(x)

    def _standardize_and_fill(self, x):
        """Standardize the continuous columns with the training statistics and
        replace the NaN by 0."""
        x[:, self.cont_cols] = (x[:, self.cont_cols] - self.mean) / self.std
        x[np.isnan(x)] = 0
        return x

    def _add_interactions_and_intercept(self, x):
        """Build the final matrix [1, x, standardized products of the pairs].

        The matrix is allocated once to limit the memory usage.
        """
        n, d = x.shape
        out = np.empty((n, 1 + d + len(self.pairs)))
        out[:, 0] = 1
        out[:, 1 : d + 1] = x
        for k, (a, b) in enumerate(self.pairs):
            out[:, d + 1 + k] = (
                x[:, a] * x[:, b] - self.inter_mean[k]
            ) / self.inter_std[k]
        return out


def find_code_rules(x):
    """Find how to decode the special answer codes of each column.

    In the BRFSS survey, the answers "don't know" and "refused" are coded with
    7 and 9 (or 77/99, 777/999, ...), "none" with 8 (88, 888) and quantities
    are often given as 1xx = per day/week, 2xx = per week/month, ... The rule
    of a column is deduced from its largest value M on the training set:

    * M = 9 (answers 1, 2, ...): 9 -> NaN, and 7 -> NaN unless 6 is also a
      valid answer (then 7 is a normal answer);
    * M = 99: 77, 99 -> NaN and 88 ("none") -> 0;
    * M = 999: 777, 999 -> NaN and 888, 555 ("none", "never") -> 0; if all
      values are >= 101 the column is a frequency "Pxx" (P = period) and is
      converted to a number of times per day;
    * M = 900, 9999, 99900 or 999999: M (and 7777, 777777) -> NaN; for
      M = 9999 (weight, height), the values >= 9000 (metric units) too.

    Args:
        x (np.ndarray): raw data, shape (N, D).

    Returns:
        dict: for each column index with a rule, a dict with the keys
            'missing' (codes -> NaN), 'missing_from' (values >= it -> NaN),
            'zero' (codes -> 0) and 'period' (None, or the number of days of
            each period 1xx, 2xx, ... to get a number per day).
    """
    rules = {}
    for j in range(x.shape[1]):
        values = np.unique(x[:, j][~np.isnan(x[:, j])])
        if len(values) == 0:
            continue
        m = values.max()
        rule = {"missing": [], "zero": [], "period": None, "missing_from": np.inf}
        if m == 9:
            rule["missing"] = [9] if 6 in values else [7, 9]
        elif m == 99:
            rule["missing"], rule["zero"] = [77, 99], [88]
        elif m == 999:
            rule["missing"], rule["zero"] = [777, 999], [555, 888]
            if values.min() >= 101:
                # 1xx/2xx/3xx/4xx = per day/week/month/year if a 3xx code
                # exists (food, blood sugar), otherwise per week/month.
                has_3xx = np.any((values >= 300) & (values < 400))
                rule["period"] = [1, 7, 30, 365] if has_3xx else [7, 30]
        elif m in (900, 9999, 99900, 999999):
            rule["missing"] = [m, 7777, 777777]
            if m == 9999:
                rule["missing_from"] = 9000
        else:
            continue
        rules[j] = rule
    return rules


def decode_special_codes(x, rules):
    """Decode the special answer codes with the rules of ``find_code_rules``.

    Args:
        x (np.ndarray): raw data, shape (N, D).
        rules (dict): output of ``find_code_rules``.

    Returns:
        np.ndarray: decoded data, shape (N, D).
    """
    if not rules:
        return x
    x = x.copy()
    for j, rule in rules.items():
        col = x[:, j]
        missing = np.isin(col, rule["missing"]) | (col >= rule["missing_from"])
        zero = np.isin(col, rule["zero"])
        if rule["period"] is not None:
            # Pxx -> xx / (number of days of period P); e.g. 203 (3 times a
            # week) -> 3/7. The code 300 ("less than once a month") gives 0.
            valid = ~np.isnan(col) & ~missing & ~zero
            period = np.zeros(len(col), dtype=int)
            period[valid] = (col[valid] // 100).astype(int)
            divisors = np.array([np.inf] + list(rule["period"]) + [np.inf] * 5)
            col = np.where(valid, (col % 100) / divisors[period], col)
        col = np.where(missing, np.nan, col)
        x[:, j] = np.where(zero, 0, col)
    return x


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


def find_interaction_pairs(x, y, k):
    """Choose the pairs of columns whose products are added as features.

    The k columns with the largest absolute correlation with the label are
    selected, and all the pairs among them are returned.

    Args:
        x (np.ndarray): preprocessed data without NaN, shape (N, D).
        y (np.ndarray): labels, shape (N,).
        k (int): number of columns to combine.

    Returns:
        list of tuple: the k * (k - 1) / 2 pairs of column indices (a, b).
    """
    # computed without building a centered copy of x (memory)
    n = len(y)
    yc = y - y.mean()
    mean = x.mean(axis=0)
    norm_xc = np.sqrt(np.maximum(np.einsum("ij,ij->j", x, x) - n * mean**2, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(x.T @ yc) / (norm_xc * np.linalg.norm(yc))
    top = np.sort(np.argsort(-np.nan_to_num(corr))[:k])
    return [(a, b) for i, a in enumerate(top) for b in top[i + 1 :]]
