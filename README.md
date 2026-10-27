# ML Project 1 — Cardiovascular disease prediction

Predicting MICHD (coronary heart disease / myocardial infarction) from the BRFSS 2015
survey, with the 6 methods of `source/implementations.py` (numpy only). The task is
described in `project1_description.pdf`.

## Layout

```
project1_description.pdf   the assignment
report/                    the report
source/
  implementations.py       the 6 required methods (least squares, ridge, GD, SGD, logistic, reg. logistic)
  helpers.py               CSV loading, mini-batches for SGD, submission writer
  normalize.py             BRFSS special codes (CDC codebook) -> NaN / 0, units
  preprocessing.py         cleaning and imputation
  features.py              feature construction used by the report
  model.py                 F1 score and F1-maximizing threshold
  cross_validation.py      model selection and evaluation of the report (~15 min)
  run.py                   reproduces the submission (~2 min)
```

## Usage

Requires Python 3 and numpy (`pip install numpy`).

1. Download `x_train.csv`, `y_train.csv` and `x_test.csv` from AIcrowd into `data/`
   at the root of the repository.
2. Run from `source/`:

```
cd source
python run.py               # writes submission.csv
python cross_validation.py  # model comparison, lambda curve, hold-out evaluation
```
