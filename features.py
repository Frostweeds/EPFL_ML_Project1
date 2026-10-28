"""Raw BRFSS rows -> model features, the steps of the report in code order:

    Cleaning          special codes, units, identifiers dropped, missing indicators
    Quality filters   constant, incomplete, target leak
    Encoding          one-hot expansion, outlier clipping
    Selection         low output correlation, correlated pair
    Scaling           NaN -> training mean, standardization, bias

Every statistic is fitted on x_train and reused for x_test.
"""

import numpy as np

from normalize import normalize_codes, normalize_nan
from preprocessing import (
    CATEGORY_COLUMNS,
    IDENTIFIER_COLUMNS,
    LEAK_COLUMNS,
    SURVEY_DESIGN_COLUMNS,
    add_bias,
    add_missing_indicator_columns,
    expand_category_column,
    find_constant_columns,
    find_correlated_columns,
    find_incomplete_columns,
    find_low_output_correlation_columns,
    find_missing_indicator_columns,
    fit_outlier_bounds,
    standardize,
)


def drop(labels, x_train, x_test, filter_names):
    """Drop filter_names from x_train and x_test. Returns the new labels, x_train, x_test."""
    kept = [i for (i, name) in labels.items() if name not in filter_names]
    new_labels = dict(enumerate(labels[i] for i in kept))
    return new_labels, x_train[:, kept], x_test[:, kept]


def build_features(x_labels, x_train, x_test, y_train, return_names=False):
    """Returns tx_train and tx_test, bias first; optionally also the feature names."""
    # Cleaning
    labels = x_labels
    x_tr = normalize_codes(labels, x_train)
    x_te = normalize_codes(labels, x_test)
    labels, x_tr, x_te = drop(
        labels, x_tr, x_te, IDENTIFIER_COLUMNS + SURVEY_DESIGN_COLUMNS
    )
    # Before the quality filters: a dropped column keeps its indicator (e.g. STREHAB1)
    indicator_columns = find_missing_indicator_columns(labels, x_tr)
    _, x_te = add_missing_indicator_columns(labels, x_te, indicator_columns)
    labels, x_tr = add_missing_indicator_columns(labels, x_tr, indicator_columns)

    # Quality filters
    labels, x_tr, x_te = drop(labels, x_tr, x_te, find_constant_columns(labels, x_tr))
    labels, x_tr, x_te = drop(labels, x_tr, x_te, find_incomplete_columns(labels, x_tr))
    labels, x_tr, x_te = drop(labels, x_tr, x_te, LEAK_COLUMNS)

    # Encoding
    for name in CATEGORY_COLUMNS:
        j = {n: i for (i, n) in labels.items()}[name]
        categories = np.unique(x_tr[~np.isnan(x_tr[:, j]), j])
        _, x_te = expand_category_column(labels, x_te, name, categories)
        labels, x_tr = expand_category_column(labels, x_tr, name, categories)
    lower, upper = fit_outlier_bounds(x_tr)
    x_tr, x_te = np.clip(x_tr, lower, upper), np.clip(x_te, lower, upper)

    # Selection, on the encoded and clipped features: the nominal codes (MARITAL 1 - 6)
    # are only correlated with y as dummies, and a few outliers can hide a duplicate
    # (PAFREQ2_ = 7 x EXEROFT2)
    labels, x_tr, x_te = drop(
        labels, x_tr, x_te, find_low_output_correlation_columns(labels, x_tr, y_train)
    )
    labels, x_tr, x_te = drop(
        labels, x_tr, x_te, find_correlated_columns(labels, x_tr, y_train)
    )

    # Scaling
    mean, std = np.nanmean(x_tr, axis=0), np.nanstd(x_tr, axis=0)
    tx_train = add_bias(standardize(normalize_nan(x_tr, mean), mean, std))
    tx_test = add_bias(standardize(normalize_nan(x_te, mean), mean, std))
    if return_names:
        return tx_train, tx_test, np.array(["bias", *labels.values()])
    return tx_train, tx_test


def add_harehab1(tx, tx_test, raw_train, raw_test):
    """Append the raw answered indicator for the separate HAREHAB1 comparison."""
    answered, answered_test = (~np.isnan(raw_train)).astype(float), (~np.isnan(raw_test)).astype(float)
    mean, std = answered.mean(), answered.std()
    std = std if std > 1e-12 else 1.0
    return np.c_[tx, (answered - mean) / std], np.c_[tx_test, (answered_test - mean) / std]
