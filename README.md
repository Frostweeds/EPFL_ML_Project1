# EPFL ML Project 1 — MICHD Risk Prediction

Class Project 1 for [EPFL CS-433 Machine Learning](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/),
Fall 2026. Predicting the risk of coronary heart disease (MICHD) from BRFSS health-survey data,
using only NumPy implementations of the methods seen in class.

Full statement: [`project1_description.pdf`](project1_description.pdf) (copied from the
[course repo](https://github.com/epfml/ML_course/tree/main/projects/project1)).
Competition: https://www.aicrowd.com/challenges/epfl-machine-learning-project-1
Submission deadline: **Oct 29th, 2026, 16:00**.

## Repo layout

```
implementations.py      Required: the 6 graded ML methods (see below). Must stay a plain
                        .py file — the course's grading tests import it as a module.
helpers.py              Provided by the course: load_csv_data, create_csv_submission
                        (+ load_csv_data_labels: column names of x_train).
codebook.py             Special answer codes per column (don't know / refused, none / never),
                        from the BRFSS 2015 codebook.
normalize.py            Normalization: special codes -> NaN / 0, units -> kg, cm, per day, ...
preprocessing.py        Feature processing building blocks: column filters, missing indicators,
                        one-hot expansion, outlier clipping, polynomial columns, standardization.
run.py                  Reproducibility script: raw CSVs -> submission .csv (see below).
build_dataset.py        Rebuilds the datasets data/experiment_X/ from the raw CSVs (see below).
notebooks/
  build_dataset.ipynb   Builds a dataset (data/experiment_X/) by composing normalize.py and
                        preprocessing.py step by step. Settings at the top select the experiment.
  train_evaluate.ipynb  Trains / evaluates models on a dataset: the 6 methods, grid search for
                        reg_logistic_regression, cross-validated comparison of the datasets.
  project1.ipynb        First exploratory notebook (EDA, plots of the raw statistics).
data/                   x_train.csv, y_train.csv, x_test.csv (gitignored, see data/README.md)
                        and the built datasets data/experiment_X/ (gitignored).
  experiment_6.npz      The best dataset, compressed (Git LFS, see "Datasets" below).
report/                 LaTeX template (IEEEtran) for the 2-page report, from the course repo.
grading_tests/          Local copy of the course's public pytest checks (gitignored).
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only NumPy + the Python standard library may be used in the actual ML code; matplotlib is for
plots only. No pandas / scikit-learn / PyTorch / TensorFlow anywhere.

## Datasets (`data/experiment_X/`)

Every experiment is a dataset `data/experiment_X/` (`x_train.npy`, `x_test.npy`, `y_train.npy`,
bias column first), read by `notebooks/train_evaluate.ipynb`. Only the best one,
`experiment_6`, is versioned, as a compressed archive `data/experiment_6.npz` (126 MB, Git LFS);
the others (0.5 – 1.3 GB each) are rebuilt from the raw CSVs.

```bash
git lfs install && git lfs pull      # fetch data/experiment_6.npz (once, after cloning)
python build_dataset.py --unpack 6   # data/experiment_6.npz -> data/experiment_6/*.npy (seconds)

python build_dataset.py 5            # rebuild data/experiment_5/ from data/*.csv (~2 min)
python build_dataset.py 1 2 3        # several experiments
python build_dataset.py all          # every experiment (~6.5 GB of disk)
python build_dataset.py --pack 6     # data/experiment_6/ -> data/experiment_6.npz (lossless)
```

The settings of each experiment are in `EXPERIMENTS` at the top of `build_dataset.py`. The script
runs the same steps as `notebooks/build_dataset.ipynb` (use the notebook to inspect a step or try a
new setting) and was checked to reproduce the datasets used in the results below: identical for
`experiment_1` – `experiment_3` and `experiment_5` – `experiment_7`, within 1e-11 for
`experiment_4`. To add an experiment, add an entry to `EXPERIMENTS` and a row to the table below.

## Reproducing the submission

```bash
python run.py                 # experiment_5: without the HAREHAB1 leak feature
python run.py --leak          # experiment_6: with the HAREHAB1 leak feature
python run.py --leak-only     # experiment_7: bias + HAREHAB1 leak feature only
python run.py --output x.csv  # other output path (default: submission.csv)
```

~2 minutes from the raw CSVs in `data/`. `run.py` repeats the steps of
`notebooks/build_dataset.ipynb` (checked to give identical features to `data/experiment_5` and
`data/experiment_6`), trains `logistic_regression` (`gamma = 1`, `max_iters = 300`) on the
whole training set and predicts positive when the score is above the threshold maximizing F1
on the training set.

## Required methods (`implementations.py`)

All 6 functions return `(w, loss)` — the **last** weight vector (not the full history) and its
loss (for `ridge_regression` / `reg_logistic_regression`, the loss excludes the penalty term).
Vectors are 1D arrays, shape `(D,)`. All 6 pass the course's public grading tests.

| Function | Description |
|---|---|
| `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)` | Linear regression, gradient descent |
| `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)` | Linear regression, SGD (batch size 1) |
| `least_squares(y, tx)` | Least squares via normal equations |
| `ridge_regression(y, tx, lambda_)` | Ridge regression via normal equations |
| `logistic_regression(y, tx, initial_w, max_iters, gamma)` | Logistic regression, GD (`y` in `{0,1}`) |
| `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` | Regularized logistic regression, GD |

## Data preparation (`notebooks/build_dataset.ipynb`)

Every statistic (columns to drop, categories, outlier bounds, mean, std) is fitted on `x_train`
only and reused unchanged for `x_test`.

1. **Normalization** (`normalize.py`): "don't know / refused / missing" codes -> NaN, "none / never"
   codes -> 0 (per column, from `codebook.py`); weight -> kg, height -> cm, frequencies -> times
   per day, durations -> days / minutes, month-year -> fractional year.
2. **Identifier and survey design columns dropped**: dates, IDs, phone screening, survey weights,
   questionnaire version (`IDENTIFIER_COLUMNS`, `SURVEY_DESIGN_COLUMNS`).
3. **Constant columns dropped**: most frequent value in >= 99% of the non-NaN rows.
4. **Incomplete columns dropped**: NaN ratio > `INCOMPLETE_NAN_RATIO`.
5. **Missing indicators**: columns with > 5% NaN get a 0/1 `column_missing` column (NaN is often a
   skip pattern of the survey, and is predictive).
6. **Correlated features dropped**: per pair with |correlation| >= 0.95, the one less correlated
   with the output.
7. **Weakly predictive features dropped**: |correlation with y| < 0.01.
8. **One-hot expansion** of 24 nominal columns (`CATEGORY_COLUMNS`), e.g. marital status, employment,
   race; ordinal scales (general health, income, ...) stay numeric.
9. **Leak feature** (optional, `LEAK_COLUMNS`, see below).
10. **Outliers clipped** to the 1st / 99th percentiles (continuous columns only, > 10 distinct values).
11. **NaN -> train mean**, standardization, optional polynomial columns, bias column.

Labels: `y` in {-1, 1} is saved; logistic regression uses `(y == 1)` in {0, 1}.

## Evaluation protocol (`notebooks/train_evaluate.ipynb`)

- Fixed random 80 / 20 train / validation split (seed 1). Model selection and dataset
  comparisons use a **3-fold cross-validation on the 80% train split**; the validation split is
  only used for a final check.
- Metric: **F1 on the positive class** (as on AIcrowd). The classes are imbalanced (8.8% positive),
  so the decision threshold is tuned to maximize F1 on the training folds, then applied unchanged
  to the held-out fold. All datasets have the same rows, so the folds are the same for every
  experiment and differences can be compared fold by fold (paired).
- Held-out log-loss is reported as a second, less noisy metric.

## Experiments

Cross-validated F1 (3 folds, `logistic_regression`, `gamma = 1`, 300 iterations) unless stated.

| Dataset | Change | Features | CV F1 per fold | Mean CV F1 |
|---|---|---|---|---|
| `experiment_1` | Baseline: normalization, filters, mean imputation | 145 | 0.4258 / 0.4232 / 0.4323 | 0.4271 |
| `experiment_2` | + one-hot of nominal columns, `QSTVER` dropped | 246 | 0.4274 / 0.4227 / 0.4350 | 0.4284 |
| `experiment_3` | + incomplete threshold 0.9 -> 0.99, missing indicators | 358 | 0.4277 / 0.4260 / 0.4374 | 0.4304 |
| `experiment_4` | + squares of age, physical health days, alcohol, exercise ¹ | 362 | 0.4298 / 0.4264 / 0.4343 | 0.4302 |
| `experiment_5` | = `experiment_3`, rebuilt by the current notebook | 358 | 0.4277 / 0.4260 / 0.4374 | 0.4304 |
| `experiment_6` | `experiment_5` + `HAREHAB1` leak feature | 359 | 0.4358 / 0.4358 / 0.4448 | **0.4388** |
| `experiment_7` | bias + `HAREHAB1` leak feature only | 2 | 0.1607 / 0.1595 / 0.1656 | 0.1619 ² |
| `experiment_8` | `experiment_5` with missing indicators before the constant / incomplete filters (includes `HAREHAB1_missing`) | 368 | 0.4360 / 0.4363 / 0.4447 | 0.4390 ³ |
| `experiment_9` | `experiment_8` with the correlated-pair filter after the one-hot expansion | 352 | 0.4364 / 0.4364 / 0.4451 | 0.4393 ⁴ |
| `experiment_10` | `experiment_9` with the outliers clipped before the correlated-pair filter | 351 | 0.4363 / 0.4363 / 0.4451 | 0.4392 ⁵ |
| `experiment_11` | `experiment_10` without the `HAREHAB1_missing` leak | 350 | 0.4287 / 0.4272 / 0.4375 | 0.4311 |
| `experiment_12` | `experiment_11` with the low output correlation filter after one-hot and clipping | 312 | 0.4275 / 0.4256 / 0.4377 | 0.4303 ⁶ |
| `experiment_13` | `experiment_12` with the codes and nominal columns checked against the codebook PDF — the dataset described in `report/` (`report/code`) | 356 | 0.4287 / 0.4256 / 0.4368 | 0.4304 ⁷ |

On the validation split: `experiment_5` F1 = 0.4275, `experiment_6` F1 = 0.4365.

¹ `experiment_4` was built by an older version of the notebook that also re-standardized every
column after filling NaN (neutral: 0.4306 on `experiment_3` alone).
`build_dataset.py` reproduces this with `restandardize = True`.

² = the "everyone positive" baseline (2 · 0.088 / 1.088 = 0.162): the leak alone catches only 2.7% of
the positives, so the best threshold predicts every row positive. The leak is only useful on top of
the other features.

³ = `experiment_6` within noise (+0.0002), without a hand-picked leak column: the indicators of
columns later dropped as constant / incomplete are kept (if >= 100 rows are missing, never dropped as
constant). Without `HAREHAB1_missing`: 0.4282 / 0.4268 / 0.4370, mean 0.4307 (`experiment_5`: 0.4304).

⁴ = `experiment_8` within noise with 16 fewer features: the filter also drops 18 duplicate dummies
(race), leaving 1 pair with |correlation| >= 0.95 (27 in `experiment_8`). Without `HAREHAB1_missing`:
0.4288 / 0.4271 / 0.4370, mean 0.4311. `tx.T @ tx` is still singular (12 exact linear dependencies
between derived columns, e.g. `_RFSEAT3` = a sum of `SEATBELT` dummies, or a one-hot group + its
missing indicator = bias): `least_squares` max |w| 6e6 -> 148, but not well-posed; use ridge.

⁵ = `experiment_9` within noise. The pair left in `experiment_9` is `PAFREQ2_` = 7 x `EXEROFT2`:
|correlation| 0.9485 before clipping (a few capped outliers), 0.9994 after. Clipped first, the filter
drops `EXEROFT2`: no feature pair with |correlation| >= 0.95 (max 0.947). Without `HAREHAB1_missing`:
0.4287 / 0.4272 / 0.4375, mean 0.4311 (= `experiment_11`).

⁶ = `experiment_11` within noise (-0.0008), chosen for the report: the pipeline runs in contiguous
phases (cleaning, quality filters on the raw columns, encoding, selection on the encoded features,
scaling). The low output filter no longer judges the arbitrary codes of nominal columns
(MARITAL 1 - 6) but each dummy: it drops 45 weak dummies (e.g. `MARITAL_4`). 6 exact linear
dependencies left in `tx.T @ tx` (12 in `experiment_11`).

⁷ = `experiment_12` within noise, with every special code compared to the codebook text: `CHKHEMO3`
98 (never heard of the A1C test) and `SCNTWRK1` / `SCNTLWK1` 98 (zero hours) are 0, no longer 98
tests / hours; 12 more nominal columns are one-hot expanded instead of used as numbers
(`EXRACT11` / `EXRACT21` activity type, `IMFVPLAC` place of the flu shot, `_RACE`, ...,
`EYEEXAM` whose 8 = never). `report/code` is the cleaned-up code of this dataset (bit-identical).

Other experiments on `experiment_3` (no dataset saved), all **neutral or worse**:

| Experiment | Result | Interpretation |
|---|---|---|
| Methods: least squares, ridge, GD, SGD, logistic, reg. logistic | logistic 0.422 > ridge / LS / GD 0.412 > SGD 0.364 (validation split) | logistic regression is the best model; SGD (batch size 1) has not converged in 1000 steps |
| Grid search `reg_logistic_regression`: lambda in [0, 0.1], gamma in [0.25, 1] | all lambda <= 1e-3 equal (0.4267 – 0.4272) | no overfitting (train F1 ≈ validation F1): regularization cannot help |
| 300 -> 1000 -> 3000 iterations | 0.4304 / 0.4305 / 0.4305 | converged at 300 |
| More training data (learning curve) | 22k -> 44k -> 88k -> 175k rows: 0.4134 / 0.4240 / 0.4292 / 0.4304 | flattening: training on all rows (`run.py`) adds ~0.001 |
| NaN as its own category for 97 skip-pattern columns | 0.4300 (with gamma = 0.25: the dummies make gamma = 1 unstable) | same information as mean imputation + missing indicator |
| Second correlation filter after the one-hot expansion | 0.4302 | removes 18 exact duplicates (race dummies), no effect |
| Separate models for men and women | 0.4277 | half the data per model; the sex effect is captured by `SEX` |
| Class weights in the loss (positives x2, x5, x10.35) | 0.4300 / 0.4300 / 0.4292 | equivalent to moving the threshold, which is already tuned |
| Lighter filtering (no correlation filters; or only > 99.9% NaN / constant dropped) | 0.4316 / 0.4313 without `HAREHAB1` | the filters cost at most ~0.001 |

## The `HAREHAB1` leak

`HAREHAB1` ("Following your heart attack, did you go to any kind of outpatient rehabilitation?")
is only asked to respondents who reported a heart attack — part of the definition of the target
(MICHD = heart attack or coronary heart disease). **All 773 training rows where it is answered are
positive** (P(y = 1) = 1.000, vs 0.088 overall); it is answered for 0.24% of the rows of `x_train`
(2.7% of the positives) and 0.22% of the rows of `x_test` (238 rows). A scan of every raw column (answered, or a single
value, with P(y = 1) > 0.6) found no other such column.

It is dropped by the incomplete-column filter (99.8% NaN) in `experiment_1` – `experiment_5`.
`experiment_6` / `run.py --leak` add `HAREHAB1_answered` (1 if answered in the **raw** data, so the
15 "don't know / refused" answers also count), worth +0.008 F1 on every fold.

It is a property of the survey design, not a risk factor: a screening model could not use it.
Scores are reported with and without it.

| Submission | Command | CV F1 | AIcrowd F1 |
|---|---|---|---|
| without leak (`experiment_5`) | `python run.py` | 0.4304 | not submitted yet |
| with leak (`experiment_6`) | `python run.py --leak` | 0.4388 | not submitted yet |

## Testing against the course's grading checks

```bash
pytest grading_tests/test_project1_public.py --github_link <local-path-or-repo-url>
```

See `grading_tests/INSTRUCTIONS.md` for details. Before final submission, run `black .` and make
sure no `TODO`s remain — both are checked.

## Team

- Romain Jufer
