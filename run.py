"""Reproducibility script for EPFL ML Project 1.

Running `python run.py` regenerates the .csv submission from the raw CSVs in `data/`:
preprocessing (same steps as notebooks/build_dataset.ipynb), logistic regression on the
whole training set, decision threshold maximizing F1 on the training set.

    python run.py                 # experiment_5: without the HAREHAB1 leak feature
    python run.py --leak          # experiment_6: with the HAREHAB1 leak feature (see README.md)
    python run.py --leak-only     # experiment_7: only the bias and the HAREHAB1 leak feature
    python run.py --best          # best leaderboard submission (F1 0.453): --leak features +
                                  # ensemble of experiments/ensemble.py (~10 more minutes)
    python run.py --output x.csv  # other output path

Takes ~2 minutes.
"""

import argparse
import sys

import numpy as np

import implementations as impl
from helpers import create_csv_submission, load_csv_data, load_csv_data_labels
from normalize import normalize_codes, normalize_nan
from preprocessing import (
    CATEGORY_COLUMNS,
    IDENTIFIER_COLUMNS,
    SURVEY_DESIGN_COLUMNS,
    add_bias,
    add_missing_indicator_columns,
    compute_mean_std,
    drop_columns_by_name,
    drop_labels,
    expand_category_column,
    filter_incomplete_column,
    filter_outliers,
    find_constant_columns,
    find_correlated_columns,
    find_low_output_correlation_columns,
    standardize,
)

DATA_DIR = "data"
OUTPUT_PATH = "submission.csv"

# Preprocessing (see notebooks/build_dataset.ipynb)
INCOMPLETE_NAN_RATIO = 0.99
MISSING_INDICATOR_NAN_RATIO = 0.05
LEAK_COLUMNS = ["HAREHAB1"]

# Model (see notebooks/train_evaluate.ipynb)
MAX_ITERS = 300
GAMMA = 1.0


def drop(labels, x_train, x_test, filter_names):
    """Drop filter_names from x_train and x_test. Returns the new labels, x_train, x_test."""
    return (
        drop_labels(labels, filter_names),
        drop_columns_by_name(labels, x_train, filter_names),
        drop_columns_by_name(labels, x_test, filter_names),
    )


def build_features(x_labels, x_train, x_test, y_train, leak):
    """Raw rows -> features. Every statistic is fitted on x_train and reused for x_test.

    Returns the feature names, tx_train and tx_test (bias column first).
    """
    # Normalization: special codes and units
    labels = x_labels
    x_tr = normalize_codes(labels, x_train)
    x_te = normalize_codes(labels, x_test)

    # Column filters
    labels, x_tr, x_te = drop(
        labels, x_tr, x_te, IDENTIFIER_COLUMNS + SURVEY_DESIGN_COLUMNS
    )
    labels, x_tr, x_te = drop(labels, x_tr, x_te, find_constant_columns(labels, x_tr))
    labels, x_tr, x_te = drop(
        labels, x_tr, x_te, filter_incomplete_column(labels, x_tr, INCOMPLETE_NAN_RATIO)
    )

    indicator_columns = filter_incomplete_column(
        labels, x_tr, MISSING_INDICATOR_NAN_RATIO
    )
    _, x_te = add_missing_indicator_columns(labels, x_te, indicator_columns)
    labels, x_tr = add_missing_indicator_columns(labels, x_tr, indicator_columns)

    labels, x_tr, x_te = drop(
        labels, x_tr, x_te, find_correlated_columns(labels, x_tr, y_train)
    )
    labels, x_tr, x_te = drop(
        labels, x_tr, x_te, find_low_output_correlation_columns(labels, x_tr, y_train)
    )

    # One-hot expansion of the nominal columns
    for name in [name for name in CATEGORY_COLUMNS if name in labels.values()]:
        train_labels, x_tr, categories = expand_category_column(labels, x_tr, name)
        _, x_te, _ = expand_category_column(labels, x_te, name, categories)
        labels = train_labels

    # Leak feature: answered in the raw data
    if leak:
        raw_index = {name: i for (i, name) in x_labels.items()}
        leak_indices = [raw_index[name] for name in LEAK_COLUMNS]
        x_tr = np.c_[x_tr, ~np.isnan(x_train[:, leak_indices])]
        x_te = np.c_[x_te, ~np.isnan(x_test[:, leak_indices])]
        names = list(labels.values()) + [f"{name}_answered" for name in LEAK_COLUMNS]
        labels = dict(enumerate(names))

    # Outliers, NaN, scale, bias
    x_tr, outlier_bounds = filter_outliers(x_tr)
    x_te, _ = filter_outliers(x_te, bounds=outlier_bounds)
    mean, std = compute_mean_std(x_tr)
    tx_train = add_bias(standardize(normalize_nan(x_tr, mean), mean, std))
    tx_test = add_bias(standardize(normalize_nan(x_te, mean), mean, std))

    return ["bias"] + list(labels.values()), tx_train, tx_test


def f1_score(y01, prediction01):
    true_positive = np.sum((prediction01 == 1) & (y01 == 1))
    false_positive = np.sum((prediction01 == 1) & (y01 == 0))
    false_negative = np.sum((prediction01 == 0) & (y01 == 1))
    return 2 * true_positive / (2 * true_positive + false_positive + false_negative)


def find_best_threshold(scores, y01, n_thresholds=100):
    """Threshold maximizing F1, searched among the score quantiles."""
    thresholds = np.quantile(scores, np.linspace(0, 1, n_thresholds, endpoint=False))
    f1_per_threshold = [f1_score(y01, (scores >= t).astype(float)) for t in thresholds]
    return thresholds[int(np.argmax(f1_per_threshold))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--leak", action="store_true", help="add the HAREHAB1 leak feature"
    )
    parser.add_argument(
        "--leak-only",
        action="store_true",
        help="keep only the bias and the HAREHAB1 leak feature",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="leak features + ensemble (interactions, Nystrom RBF, MLP), best submission",
    )
    parser.add_argument("--output", default=OUTPUT_PATH, help="submission path")
    args = parser.parse_args()

    x_labels = load_csv_data_labels(DATA_DIR)
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_DIR)
    feature_names, tx_train, tx_test = build_features(
        x_labels, x_train, x_test, y_train, args.leak or args.leak_only or args.best
    )
    del x_train, x_test  # raw data no longer needed (memory)
    if args.leak_only:
        kept = [0] + [j for j, n in enumerate(feature_names) if n.endswith("_answered")]
        feature_names = [feature_names[j] for j in kept]
        tx_train, tx_test = tx_train[:, kept], tx_test[:, kept]
    print(
        f"features: {len(feature_names)} (leak: {args.leak}, leak only: {args.leak_only}, best: {args.best})"
    )

    y01 = (y_train == 1).astype(float)
    if args.best:  # average of the standardized scores of the 3 models (experiments/ensemble.py)
        sys.path.append("experiments")
        from ensemble import BEST, all_scores, combine

        train_scores, test_scores = combine(all_scores(tx_train, y01, tx_test, BEST), BEST)
    else:
        initial_w = np.zeros(tx_train.shape[1])
        w, loss = impl.logistic_regression(y01, tx_train, initial_w, MAX_ITERS, GAMMA)
        train_scores, test_scores = tx_train.dot(w), tx_test.dot(w)
        print(f"train loss {loss:.4f}")
    threshold = find_best_threshold(train_scores, y01)
    print(f"threshold {threshold:.3f}")

    y_pred = np.where(test_scores >= threshold, 1, -1)
    create_csv_submission(test_ids, y_pred, args.output)
    print(
        f"{args.output}: {len(y_pred)} predictions, {np.mean(y_pred == 1):.3f} positive"
    )


if __name__ == "__main__":
    main()
