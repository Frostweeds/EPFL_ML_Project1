# ML Project 1 — Cardiovascular disease prediction

Predicting MICHD (coronary heart disease / myocardial infarction) from the BRFSS 2015
survey. The assignment is in `project1_description.pdf`.
Training uses NumPy and the Python standard library only. The six required methods
are least squares, ridge, MSE with GD/SGD, logistic and regularized logistic regression.

## Layout

```
implementations.py         the 6 required methods, unchanged
run.py                     trains the final quadratic model and writes predictions
final_validation.py        quadratic/cubic CV and final evaluation with/without HAREHAB1
cross_validation.py        original six-method and regularization comparisons
polynomial.py              selected polynomial features and calls to the homework GD methods
features.py                raw data -> model features
normalize.py               BRFSS special codes and units
preprocessing.py           cleaning, encoding, filtering and scaling
model.py                   F1 score and decision threshold selection
helpers.py                 CSV loading and submission writer
experiments/results/final_models/   current metrics and LaTeX tables
notebooks/final_validation.ipynb    editable display of the saved tables
```

## Usage

Requires Python 3 and NumPy (`pip install numpy`). Download `x_train.csv`,
`y_train.csv` and `x_test.csv` from AIcrowd into `data/`, or pass another directory.
Run from the repository root:

```bash
python run.py data --output submission.csv
python final_validation.py data
```

The first command trains on all labeled rows and writes test predictions with columns
`Id,Prediction` and labels `-1/+1`. The second runs the complete comparison and saves
`experiments/results/final_models/results.json` and `tables.tex`. Open
`notebooks/final_validation.ipynb` to view the tables without retraining.

The evaluation can also be run in separate stages:

```bash
python final_validation.py data --fold 1   # repeat with --fold 2 and --fold 3
python final_validation.py data --holdout
python final_validation.py --collect
```

`python run.py data --holdout` evaluates only the final model.
`python cross_validation.py data` reruns the original method comparisons.
All commands above work from the raw CSVs; a local preprocessing cache is optional
through `--prepared-dir` for `run.py` and `final_validation.py`.

## Data preparation

`features.py` preserves the original preprocessing pipeline. Statistics are fitted
on the labeled dataset and reused for the competition test set:

1. Normalize missing-answer codes and units using `normalize.py`.
2. Remove identifiers and survey-design columns.
3. Add missing indicators for columns with over 5% missing values and at least 100 missing rows.
4. Remove columns that are over 99% missing or at least 99% constant, and remove the HAREHAB1 leak indicator.
5. One-hot encode nominal features and clip continuous inputs at their 1% and 99% quantiles.
6. Remove features with absolute target correlation below 0.01; for pairs correlated at least 0.95, retain the feature more correlated with the target.
7. Impute training means, standardize, and add a bias column: 356 columns including bias.

## Final model

Keep all original linear features. Fit preliminary logistic regression for 300 GD
steps at `gamma = 1`, then rank inputs by coefficient magnitude multiplied by their
training standard deviation. Select the smallest subset covering 50% of the total
magnitude, excluding the bias, and add its squares and pairwise products. Rescale
selected inputs using training min–max values before expansion, then standardize
all non-bias columns using training statistics. Redundant powers of discrete inputs
are omitted.

The final classifier is logistic regression with `lambda = 1e-3`. Initialize from
a regularized linear model trained for 1,000 steps at `gamma = 1`, then run 5,000
steps at `gamma = 0.2` on the expanded matrix. These are the unchanged homework GD
functions in `implementations.py`; the penalty is `lambda * sum(w**2)`, including
bias. Training uses fixed iteration counts and records the final gradient as a
diagnostic. Arrays remain float64; the entry points default to four BLAS threads.

Select the threshold maximizing training F1 among 100 score quantiles and apply it
unchanged to evaluation or test rows. `run.py` repeats the full procedure on all
labeled rows; its training metadata is saved in
`experiments/results/final_models/final_training.json`.

## Evaluation

Use the original seed-1 80/20 split and three folds within the 80% training portion.
Baseline preprocessing precedes this split, as in the earlier experiments; feature
selection, polynomial scaling, model fitting and threshold selection use each fold's
training rows. The 20% holdout has been evaluated previously.

`final_validation.py` produces two tables:

- Degrees 2 and 3, each with `lambda = 0` and `1e-3`: mean training F1, CV F1 with
  fold standard deviation, and CV F1 at score threshold zero. Cubic terms use only
  the ten highest-ranked inputs and retain all existing linear and quadratic terms.
- The regularized quadratic model with and without HAREHAB1: three-fold CV and final
  training/holdout F1, precision, recall, accuracy and log loss. Confusion counts,
  selected inputs, thresholds and gradients are included in the saved JSON.

The HAREHAB1 comparison appends the raw answered indicator, including refused/unknown
answers. It remains separate because this question reveals information about the
target; the default submission excludes it.
