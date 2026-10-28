# ML Project 1 — Cardiovascular disease prediction

Predicting MICHD (coronary heart disease / myocardial infarction) from the BRFSS 2015
survey, with the 6 methods of `implementations.py` (numpy only). The task is described in
`project1_description.pdf`.

## Layout

```
project1_description.pdf   the assignment
report/                    the report
implementations.py         the 6 required methods (least squares, ridge, GD, SGD, logistic, reg. logistic),
                           plus build_quadratic (degree-2 terms)
run.py                     reproduces the submission (~11 min)
cross_validation.py        model selection and evaluation of the report (~40 min)
run_experiment.py          kernel experiment: Nystrom RBF features + logistic regression (~6 min)
features.py                feature construction: raw rows -> model features
normalize.py               BRFSS special codes (CDC codebook) -> NaN / 0, units
preprocessing.py           column lists and the cleaning, encoding and selection steps
model.py                   F1 score, F1-maximizing threshold, the quadratic model
helpers.py                 CSV loading, submission writer
```

## Usage

Requires Python 3 and numpy (`pip install numpy`).

1. Download `x_train.csv`, `y_train.csv` and `x_test.csv` from AIcrowd into a folder,
   by default `data/` at the root of the repository.
2. Run from the repository root, optionally passing that folder (default `./data`):

```
python run.py [data_dir]               # writes submission.csv
python cross_validation.py [data_dir]  # model comparison, lambda curve, hold-out evaluation
python run_experiment.py [data_dir]    # kernel experiment, writes submission_kernel.csv
```

## Pipeline

### Data preparation and features (`features.py`)

Every statistic (means, quantiles, correlations, categories) is fitted on `x_train` only
and reused unchanged on `x_test`.

1. **Special codes and units** (`normalize.py`): "don't know / refused / missing" answers
   (e.g. 7, 9, 77, 99) become NaN and "none / never" answers (e.g. 88 days) become 0, per
   column, from the BRFSS 2015 codebook. Weights are converted to kg, heights to cm and
   frequencies (e.g. 203 = 3 per week) to times per day.
2. **Identifiers dropped**: state, interview date, sequence number and survey-design
   columns (weights, questionnaire version).
3. **Missing indicators**: a 0/1 column `X_missing` for each column with more than 5% NaN
   (and at least 100 NaN rows).
4. **Constant / empty columns dropped**: one value in at least 99% of the non-NaN rows
   (missing indicators excepted), or more than 99% NaN; plus `HAREHAB1_missing`, which leaks
   the target (cardiac rehabilitation is only asked after a heart attack).
5. **One-hot encoding** of the nominal columns (`CATEGORY_COLUMNS` in `preprocessing.py`:
   marital status, employment, diabetes, ...), one 0/1 column per category.
6. **Outlier clipping**: continuous columns (more than 10 distinct values) clipped to their
   1% and 99% quantiles.
7. **Feature selection**: drop the columns with |correlation with y| < 0.01, then, for every
   pair of features with |correlation| >= 0.95, the one less correlated with y.
8. **Scaling**: NaN -> training mean, standardization, bias column. 356 features in the end.

### Model (`run.py`, `quadratic_logistic` in `model.py`)

Logistic regression on the 356 features and the degree-2 terms of the strongest ones, all
fitted on the whole training set:

1. **Linear model**: logistic regression (`logistic_regression`, lambda = 0), gradient
   descent from w = 0, gamma = 1, 300 steps.
2. **Strongest features**: the ones carrying 50% of the total |w| x std (about 30).
3. **Degree-2 terms** (`build_quadratic`): their squares and pairwise products (no squares
   of binary columns), min-max scaled before the products and standardized after: about
   +500 columns.
4. **Quadratic model**: regularized logistic regression (`reg_logistic_regression`,
   lambda = 1e-3) on all the columns, gamma = 0.2, 5000 steps, from the linear weights.

The positive class is rare (8.8%), so the decision threshold on the scores is not 0 but the
one maximizing F1 on the training rows, among 100 score quantiles (`model.py`). Predictions
are written in {-1, 1}.

### Model selection and evaluation (`cross_validation.py`)

- Fixed 80 / 20 split (seed 1); the 20% hold-out split is used once, for the selected model.
- 3-fold cross-validation on the 80% split. The threshold is always chosen on the training
  folds, then applied unchanged to the held-out fold.

1. The 6 methods with fixed hyperparameters and the quadratic model, compared by CV F1.
   The quadratic model's feature selection is redone in each training fold.
2. CV F1 as a function of lambda for ridge regression and regularized logistic regression
   (lambda from 0 to 10): regularization does not help the linear model (lambda = 0).
3. The selected model (quadratic) on the hold-out split.

| method | 3-fold CV F1 |
|---|---|
| least squares | 0.417 |
| ridge regression (lambda = 1e-3) | 0.418 |
| GD on the MSE (300 steps, gamma = 1 / L) | 0.417 |
| SGD on the MSE (10 epochs, gamma = 1e-5) | 0.417 |
| SGD on the MSE (1 epoch, gamma = 1e-4) | 0.410 |
| logistic regression (300 steps, gamma = 1) | 0.430 |
| regularized logistic regression (lambda = 1e-4, 300 steps, gamma = 1) | 0.431 |
| **quadratic model** | *to be measured* |

On the hold-out split:

| model | train F1 | hold-out F1 |
|---|---|---|
| logistic regression | 0.436 | 0.428 |
| squares only (course `build_poly`) | 0.436 | 0.429 |
| **quadratic model (squares and products)** | **0.444** | **0.433** |

The squares alone add little; the pairwise products carry most of the gain.

### Kernel experiment (`run_experiment.py`)

A non-linear model with the same features: the RBF kernel
k(x, z) = exp(-g ||x - z||^2 / D), with g = 1 / median of ||x - x'||^2 / D, is approximated
with 500 random training rows as landmarks (Nystrom). The (at most) 500 standardized
features phi(x), with phi(x)^T phi(z) ~ k(x, z), are appended to the 356 features, then
logistic regression (gamma < 8 / L, at most 1000 steps) and the F1-maximizing threshold, as
in `run.py`. Same 80 / 20 split as `cross_validation.py`.

| model | hold-out F1 |
|---|---|
| logistic regression | 0.428 |
| Nystrom RBF + logistic regression | 0.426 |
| quadratic model (selected) | 0.433 |

The kernel does not improve on the linear model (train F1 0.439: it fits the training rows a
little better, without generalizing better), unlike the quadratic model.
