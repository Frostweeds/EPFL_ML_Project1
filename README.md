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
implementations.py     Required: the 6 graded ML methods (see below). Must stay a plain
                        .py file — the course's grading tests import it as a module.
helpers.py              Provided by the course, unmodified: load_csv_data, create_csv_submission.
run.py                  Reproducibility script: python run.py must regenerate the exact .csv
                        used for the team's best AIcrowd submission.
notebooks/project1.ipynb  Main notebook: EDA, feature engineering, model selection, submission.
                        Imports and calls implementations.py rather than duplicating logic.
data/                   Put x_train.csv, y_train.csv, x_test.csv here (gitignored, see data/README.md).
report/                 LaTeX template (IEEEtran) for the 2-page report, from the course repo.
grading_tests/          Local copy of the course's public pytest checks (gitignored — pulled for
                        convenience, re-fetch from the course repo before trusting it; see
                        grading_tests/INSTRUCTIONS.md).
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only NumPy + the Python standard library may be used in the actual ML code; matplotlib is for
plots only. No pandas / scikit-learn / PyTorch / TensorFlow anywhere in `implementations.py`,
`run.py`, or the notebook.

## Required methods (`implementations.py`)

All 6 functions return `(w, loss)` — the **last** weight vector (not the full history) and its
loss (for `ridge_regression` / `reg_logistic_regression`, the loss must exclude the penalty term).
Vectors are 1D arrays, shape `(D,)`.

| Function | Description | Status |
|---|---|---|
| `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)` | Linear regression, gradient descent | done — ported from `labs/ex02` |
| `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)` | Linear regression, SGD (batch size 1) | done — ported from `labs/ex02` |
| `least_squares(y, tx)` | Least squares via normal equations | TODO |
| `ridge_regression(y, tx, lambda_)` | Ridge regression via normal equations | TODO |
| `logistic_regression(y, tx, initial_w, max_iters, gamma)` | Logistic regression, GD (`y` in `{0,1}`) | TODO |
| `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` | Regularized logistic regression, GD | TODO |

The first two were ported from `../labs/ex02/template/ex02.ipynb` (your completed lab work,
still uncommitted there) and verified against the course's `grading_tests` expected values.
The other 4 aren't covered by any released lab yet (only `ex01`/`ex02` exist in the course repo
so far) — they're stubbed with `NotImplementedError`; fill them in using the shared helpers
`sigmoid`, `compute_logistic_loss`, `compute_logistic_gradient` at the top of the file (`compute_mse`
and `compute_mse_gradient` are already implemented and reused by the two done methods).
`batch_iter` lives in `helpers.py` (imported by `implementations.py`), since it's generic to any
mini-batch method rather than specific to one implementation.

## Workflow

1. Download the dataset (see `data/README.md`).
2. Implement the 6 methods in `implementations.py`.
3. Work through `notebooks/project1.ipynb`: EDA → feature processing → baselines → cross-validated
   model selection → final predictions.
4. Once a final pipeline is chosen, mirror it in `run.py` so it's independently reproducible.
5. Write the report in `report/latex-template.tex` (max 2 pages + 1 page references, no appendix).

## Testing against the course's grading checks

```bash
# from a clone of https://github.com/epfml/ML_course, in projects/project1/grading_tests/
pytest --github_link <local-path-or-repo-url> .
```

See `grading_tests/INSTRUCTIONS.md` for details (conda env, etc.). Before final submission, run
`black .` and make sure no `TODO`s remain — both are checked.

## Team

- Romain Jufer
