# CS-433 Machine Learning – Project 1: predicting coronary heart disease (MICHD)

Team: Yvan Bénon, Grégoire Taillard, Jean Erard

We predict whether a person has coronary heart disease (MICHD) from the answers
to the BRFSS health survey (328,135 training samples, 321 features, about 9% of
positive samples). Our model is a class-weighted, L2-regularized logistic
regression trained with Newton's method on cleaned features with pairwise
interactions, with a decision threshold tuned to maximize the F1 score
(AIcrowd: F1 = 0.443, accuracy = 0.882).

## Repository structure

| File | Content |
| --- | --- |
| `implementations.py` | The six required methods (`mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression`) and their losses/gradients. |
| `preprocessing.py` | `Preprocessor` class: preprocessing learned on the training set and applied identically to the test set. |
| `models.py` | Our model (weighted regularized logistic regression with Newton's method) and the metrics (accuracy, F1). |
| `cross_validation.py` | K-fold cross-validation for the regularization parameter and the decision threshold. |
| `helpers.py` | Data loading and creation of the submission file. |
| `plots.py` | Plotting functions (visualization only). |
| `run.py` | Reproduces our best submission. |
| `exploration.ipynb` | Data exploration, baselines with the six methods, hyperparameter search and ablation study. |
| `report/` | Our 2-page report (`report.pdf`) and its LaTeX source (`report.tex`, compile with `pdflatex report.tex`). |

## How to run

Requirements: Python 3 with NumPy (and Matplotlib for the notebook only). No
other library is used. About 6 GB of RAM are needed.

1. Download the data from AIcrowd and put `x_train.csv`, `y_train.csv` and
   `x_test.csv` in a folder `data/` at the root of the repository (this folder
   is ignored by git).
2. From the root of the repository, run:

   ```bash
   python run.py          # uses the hyperparameters found by cross-validation (~5 min)
   python run.py --tune   # redoes the cross-validation search first (~45 min)
   ```

   The predictions are written to `submission.csv`.

## Method

### Preprocessing (`preprocessing.py`)

All statistics are computed on the training set only, then applied to the test set.

1. Drop the survey metadata (interview date, sample ids, sampling weights).
2. Decode the special BRFSS answer codes, with a rule deduced from the largest value of each feature:
   7/9, 77/99, 777/999 ("don't know", "refused") become missing values;
   8/88/888 ("none") and 555 ("never") become 0;
   frequencies coded `Pxx` (e.g. `203` = 3 times per week) are converted to a number of times per day;
   9999-type codes and metric units of weight/height become missing values.
3. Drop the features with at least 95% of missing values.
4. Drop the (almost) constant features: most frequent value in at least 99% of the non-missing values.
5. One-hot encode the non-binary features with at most 10 distinct values (missing value counted as a value): one indicator per category except the last one (reference), plus one indicator for missing values when needed.
6. Standardize the other features (mean 0, standard deviation 1, ignoring missing values).
7. Replace the remaining missing values by 0, the mean after standardization.
8. Drop the features whose absolute correlation with an earlier feature is at least 0.99.
9. Add the standardized products of all pairs among the 30 features most correlated with the label (interactions such as sex × age).
10. Add an intercept column.

This gives 867 features.

### Model (`models.py`)

* Logistic regression with an L2 penalty `lambda * ||w||^2`, labels in {-1, 1}.
* **Balanced class weights**: each sample is weighted by `N / (2 * N_class)` to compensate for the class imbalance.
* Trained with **Newton's method** (15 iterations), which converges much faster than gradient descent and has no step size to tune.
* Prediction: 1 if `x^T w >= threshold`, -1 otherwise.

### Hyperparameters (`cross_validation.py`, `run.py --tune`)

The search uses 60% of the training set (random split, seed 42); the other 40%
is a hold-out set used only to evaluate the chosen model. For each `lambda` in
`{1e-4, 3.2e-4, 1e-3, 3.2e-3, 1e-2}` and each threshold in `-2, -1.9, ..., 1.9`,
the mean F1 score is computed by 5-fold cross-validation; the best pair is
`lambda = 1e-3`, threshold `= 1.1` (cross-validation F1 = 0.425).
The final model is then retrained on the whole training set with these values.

## Results

Ablation study on the hold-out set (threshold tuned by 5-fold cross-validation
on the training part; see the last section of `exploration.ipynb`):

| Variant | Features | Threshold | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| Old preprocessing (first submission, `lambda = 2.15e-5`) | 417 | 0.9 | 0.433 | 0.866 |
| + metadata removed, codes decoded | 332 | 1.0 | 0.434 | 0.874 |
| + one-hot encoding up to 10 categories | 441 | 0.9 | 0.433 | 0.866 |
| + 30 interactions (final) | 866 | 1.1 | 0.432 | 0.878 |
| Final without class weights | 866 | -1.3 | 0.435 | 0.876 |
| Final without threshold tuning | 866 | 0.0 | 0.378 | 0.760 |

The differences between the preprocessing variants are within the noise of
the hold-out estimate (about ±0.004). In a paired 5-fold comparison on the full
training set (same folds, best threshold of each fold), the final
preprocessing is better than the old one on all 5 folds (+0.004 F1 on
average, +0.0016 without the interactions). Tuning the decision threshold is
by far the most important ingredient (+0.055 F1); class weights and threshold
tuning are almost interchangeable.

Other ideas that did not improve the cross-validation F1 score: quantile
binning of the continuous features, clipping of the outliers, missing-value
indicators for the continuous features, more one-hot categories (15).

To train without interactions, set `N_INTERACTIONS = 0` and
`BEST_THRESHOLD = 0.9` in `run.py`.

**AIcrowd results:** first submission (old preprocessing) F1 = 0.438,
accuracy = 0.869; final model (`run.py`) **F1 = 0.443, accuracy = 0.882**.
