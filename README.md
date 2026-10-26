# CS-433 Machine Learning – Project 1: predicting coronary heart disease (MICHD)

Team: Yvan Bénon, Grégoire Taillard, Jean Erard

We predict whether a person has coronary heart disease (MICHD) from the answers
to the BRFSS health survey (328,135 training samples, 321 features, about 9% of
positive samples). Our model is a class-weighted, L2-regularized logistic
regression trained with Newton's method, with a decision threshold tuned to
maximize the F1 score.

**AIcrowd result: F1 = 0.438, accuracy = 0.869.**

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
| `exploration.ipynb` | Data exploration, baselines with the six methods, and hyperparameter search. |
| `report/` | Our 2-page report (`report.pdf`) and its LaTeX source (`report.tex`, compile with `pdflatex report.tex`). |

## How to run

Requirements: Python 3 with NumPy (and Matplotlib for the notebook only). No
other library is used.

1. Download the data from AIcrowd and put `x_train.csv`, `y_train.csv` and
   `x_test.csv` in a folder `data/` at the root of the repository (this folder
   is ignored by git).
2. From the root of the repository, run:

   ```bash
   python run.py          # uses the hyperparameters found by cross-validation
   python run.py --tune   # redoes the cross-validation search first (slower)
   ```

   The predictions are written to `submission.csv`.

## Method

### Preprocessing (`preprocessing.py`)

All statistics are computed on the training set only, then applied to the test set.

1. Drop the features with at least 95% of missing values.
2. Drop the (almost) constant features: most frequent value in at least 99% of the non-missing values.
3. One-hot encode the non-binary features with at most 5 distinct values (missing value counted as a value): one indicator per category except the last one (reference), plus one indicator for missing values when needed.
4. Standardize the other features (mean 0, standard deviation 1, ignoring missing values).
5. Replace the remaining missing values by 0, the mean after standardization.
6. Drop the features whose absolute correlation with an earlier feature is at least 0.99.
7. Add an intercept column.

### Model (`models.py`)

* Logistic regression with an L2 penalty `lambda * ||w||^2`, labels in {-1, 1}.
* **Balanced class weights**: each sample is weighted by `N / (2 * N_class)` to compensate for the class imbalance.
* Trained with **Newton's method** (15 iterations), which converges much faster than gradient descent and has no step size to tune.
* Prediction: 1 if `x^T w >= threshold`, -1 otherwise.

### Hyperparameters (`cross_validation.py`, `run.py --tune`)

The search uses 60% of the training set (random split, seed 42); the other 40%
is a hold-out set used only to evaluate the chosen model.

* `lambda` is chosen among 10 values in `[1e-8, 1e-3]` (log scale) by 5-fold
  cross-validation, minimizing the validation logistic loss.
* The threshold is chosen among `-2, -1.9, ..., 1.9` by 5-fold cross-validation,
  maximizing the validation F1 score.

The final model is then retrained on the whole training set with these values.
