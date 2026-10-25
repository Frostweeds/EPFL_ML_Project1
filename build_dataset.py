"""Build the datasets data/experiment_X/ from the raw CSVs in data/.

Same steps as notebooks/build_dataset.ipynb, with the settings of each experiment in EXPERIMENTS
(see README.md for their results). Writes x_train.npy, x_test.npy, y_train.npy (the format read by
notebooks/train_evaluate.ipynb).

    python build_dataset.py 5                # build data/experiment_5/
    python build_dataset.py 1 2 3            # several experiments
    python build_dataset.py all              # every experiment
    python build_dataset.py --unpack 6       # data/experiment_6.npz (versioned, Git LFS) -> data/experiment_6/
    python build_dataset.py --pack 6         # data/experiment_6/ -> data/experiment_6.npz

Each build takes ~2 minutes and 0.5 - 1.3 GB of disk.
"""

import argparse
import os

import numpy as np

from helpers import load_csv_data, load_csv_data_labels
from normalize import normalize_codes, normalize_nan, replace_codes
from preprocessing import (
    CATEGORY_COLUMNS,
    IDENTIFIER_COLUMNS,
    MIN_ROWS,
    SURVEY_DESIGN_COLUMNS,
    add_bias,
    add_missing_indicator_columns,
    add_polynomial_columns,
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

DEFAULT_SETTINGS = {
    "keep_columns": [],  # survey design columns kept anyway
    "expand_categories": True,
    "incomplete_nan_ratio": 0.99,  # columns with more NaN are dropped
    "missing_indicator_nan_ratio": 0.05,  # None: no missing indicators
    "early_missing_indicators": False,  # before the constant / incomplete filters
    "correlation_after_one_hot": False,  # feature - feature filter also sees the dummies
    "clip_before_correlation": False,  # the late filter sees the clipped values
    "drop_columns": [],  # dropped once the missing indicators exist
    "low_output_after_one_hot": False,  # judges each dummy, not the nominal codes
    "codebook_fixes": False,  # CODEBOOK_FIX_NONE_CODES, EXTRA_CATEGORY_COLUMNS
    "poly_columns": [],
    "poly_degree": 2,
    "leak_columns": [],
    "leak_only": False,  # keep only the bias and the leak features
    "restandardize": False,  # second standardization of every column (experiment_4)
}

EXPERIMENTS = {
    1: {
        "keep_columns": ["QSTVER"],
        "expand_categories": False,
        "incomplete_nan_ratio": 0.9,
        "missing_indicator_nan_ratio": None,
    },
    2: {"incomplete_nan_ratio": 0.9, "missing_indicator_nan_ratio": None},
    3: {},
    4: {
        "poly_columns": ["_AGEG5YR", "PHYSHLTH", "DROCDY3_", "EXERHMM1"],
        "restandardize": True,
    },
    5: {},
    6: {"leak_columns": ["HAREHAB1"]},
    7: {"leak_columns": ["HAREHAB1"], "leak_only": True},
    8: {"early_missing_indicators": True},
    9: {"early_missing_indicators": True, "correlation_after_one_hot": True},
    10: {
        "early_missing_indicators": True,
        "correlation_after_one_hot": True,
        "clip_before_correlation": True,
    },
    # experiment_10 without the HAREHAB1 leak
    11: {
        "early_missing_indicators": True,
        "correlation_after_one_hot": True,
        "clip_before_correlation": True,
        "drop_columns": ["HAREHAB1_missing"],
    },
    # experiment_11 with the low output correlation filter after one-hot and clipping
    12: {
        "early_missing_indicators": True,
        "correlation_after_one_hot": True,
        "clip_before_correlation": True,
        "drop_columns": ["HAREHAB1_missing"],
        "low_output_after_one_hot": True,
    },
    # experiment_12 with the codes and nominal columns checked against the codebook PDF
    13: {
        "early_missing_indicators": True,
        "correlation_after_one_hot": True,
        "clip_before_correlation": True,
        "drop_columns": ["HAREHAB1_missing"],
        "low_output_after_one_hot": True,
        "codebook_fixes": True,
    },
}

# Missed by codebook.py: 98 = never heard of the A1C test, zero hours of work
CODEBOOK_FIX_NONE_CODES = {"CHKHEMO3": [98], "SCNTWRK1": [98], "SCNTLWK1": [98]}

# Nominal columns missed by CATEGORY_COLUMNS
EXTRA_CATEGORY_COLUMNS = [
    "EYEEXAM",
    "EXRACT11",
    "EXRACT21",
    "IMFVPLAC",
    "CRGVREL1",
    "CRGVPRB1",
    "TETANUS",
    "ASPUNSAF",
    "SCNTPAID",
    "SCNTLPAD",
    "_CPRACE",
    "_RACE",
]


def drop(labels, x_train, x_test, filter_names):
    """Drop filter_names from x_train and x_test. Returns the new labels, x_train, x_test."""
    return (
        drop_labels(labels, filter_names),
        drop_columns_by_name(labels, x_train, filter_names),
        drop_columns_by_name(labels, x_test, filter_names),
    )


def build_features(x_labels, x_train, x_test, y_train, settings):
    """Raw rows -> features. Every statistic is fitted on x_train and reused for x_test.

    Returns the feature names, tx_train and tx_test (bias column first).
    """
    # Normalization: special codes and units
    labels = x_labels
    x_tr = normalize_codes(labels, x_train)
    x_te = normalize_codes(labels, x_test)
    if settings["codebook_fixes"]:
        x_tr = replace_codes(labels, x_tr, CODEBOOK_FIX_NONE_CODES, 0)
        x_te = replace_codes(labels, x_te, CODEBOOK_FIX_NONE_CODES, 0)

    # Column filters
    non_feature_columns = [
        name
        for name in IDENTIFIER_COLUMNS + SURVEY_DESIGN_COLUMNS
        if name not in settings["keep_columns"]
    ]
    labels, x_tr, x_te = drop(labels, x_tr, x_te, non_feature_columns)

    # Early: the indicators of the columns dropped as constant / incomplete are kept
    # (e.g. HAREHAB1_missing), if at least MIN_ROWS rows are missing
    early = settings["early_missing_indicators"]
    if early and settings["missing_indicator_nan_ratio"] is not None:
        index = {name: j for (j, name) in labels.items()}
        indicator_columns = [
            name
            for name in filter_incomplete_column(
                labels, x_tr, settings["missing_indicator_nan_ratio"]
            )
            if np.isnan(x_tr[:, index[name]]).sum() >= MIN_ROWS
        ]
        _, x_te = add_missing_indicator_columns(labels, x_te, indicator_columns)
        labels, x_tr = add_missing_indicator_columns(labels, x_tr, indicator_columns)

    # Indicators are rare by design, never dropped as constant
    constant_columns = [
        name
        for name in find_constant_columns(labels, x_tr)
        if not name.endswith("_missing")
    ]
    labels, x_tr, x_te = drop(labels, x_tr, x_te, constant_columns)
    labels, x_tr, x_te = drop(
        labels,
        x_tr,
        x_te,
        filter_incomplete_column(labels, x_tr, settings["incomplete_nan_ratio"]),
    )

    if not early and settings["missing_indicator_nan_ratio"] is not None:
        indicator_columns = filter_incomplete_column(
            labels, x_tr, settings["missing_indicator_nan_ratio"]
        )
        _, x_te = add_missing_indicator_columns(labels, x_te, indicator_columns)
        labels, x_tr = add_missing_indicator_columns(labels, x_tr, indicator_columns)

    labels, x_tr, x_te = drop(
        labels,
        x_tr,
        x_te,
        [name for name in settings["drop_columns"] if name in labels.values()],
    )

    correlation_after_one_hot = settings["correlation_after_one_hot"]
    if not correlation_after_one_hot:
        labels, x_tr, x_te = drop(
            labels, x_tr, x_te, find_correlated_columns(labels, x_tr, y_train)
        )
    low_output_after_one_hot = settings["low_output_after_one_hot"]
    if not low_output_after_one_hot:
        labels, x_tr, x_te = drop(
            labels,
            x_tr,
            x_te,
            find_low_output_correlation_columns(labels, x_tr, y_train),
        )

    # One-hot expansion of the nominal columns
    if settings["expand_categories"]:
        category_columns = CATEGORY_COLUMNS + (
            EXTRA_CATEGORY_COLUMNS if settings["codebook_fixes"] else []
        )
        for name in [name for name in category_columns if name in labels.values()]:
            train_labels, x_tr, categories = expand_category_column(labels, x_tr, name)
            _, x_te, _ = expand_category_column(labels, x_te, name, categories)
            labels = train_labels

    # Clipped first: a few outliers can hide a duplicate (PAFREQ2_ = 7 x EXEROFT2)
    clip_early = correlation_after_one_hot and settings["clip_before_correlation"]
    if clip_early:
        x_tr, outlier_bounds = filter_outliers(x_tr)
        x_te, _ = filter_outliers(x_te, bounds=outlier_bounds)

    # The nominal codes (MARITAL 1 - 6) are labels: correlated with y only as dummies
    if low_output_after_one_hot:
        labels, x_tr, x_te = drop(
            labels,
            x_tr,
            x_te,
            find_low_output_correlation_columns(labels, x_tr, y_train),
        )

    # After: no feature pair above the threshold, dummies included (exact duplicates
    # make tx.T @ tx singular for least_squares)
    if correlation_after_one_hot:
        labels, x_tr, x_te = drop(
            labels, x_tr, x_te, find_correlated_columns(labels, x_tr, y_train)
        )

    # Leak features: answered in the raw data
    leak_columns = settings["leak_columns"]
    if leak_columns:
        raw_index = {name: i for (i, name) in x_labels.items()}
        leak_indices = [raw_index[name] for name in leak_columns]
        x_tr = np.c_[x_tr, ~np.isnan(x_train[:, leak_indices])]
        x_te = np.c_[x_te, ~np.isnan(x_test[:, leak_indices])]
        names = list(labels.values()) + [f"{name}_answered" for name in leak_columns]
        labels = dict(enumerate(names))

    # Outliers, NaN, scale
    if not clip_early:
        x_tr, outlier_bounds = filter_outliers(x_tr)
        x_te, _ = filter_outliers(x_te, bounds=outlier_bounds)
    mean, std = compute_mean_std(x_tr)
    x_tr = standardize(normalize_nan(x_tr, mean), mean, std)
    x_te = standardize(normalize_nan(x_te, mean), mean, std)

    # Polynomial columns, standardized themselves
    poly_columns = [
        name for name in settings["poly_columns"] if name in labels.values()
    ]
    n_columns = x_tr.shape[1]
    degree = settings["poly_degree"]
    _, x_te = add_polynomial_columns(labels, x_te, poly_columns, degree)
    labels, x_tr = add_polynomial_columns(labels, x_tr, poly_columns, degree)
    poly_mean, poly_std = compute_mean_std(x_tr[:, n_columns:])
    x_tr[:, n_columns:] = standardize(x_tr[:, n_columns:], poly_mean, poly_std)
    x_te[:, n_columns:] = standardize(x_te[:, n_columns:], poly_mean, poly_std)

    if settings["restandardize"]:
        mean, std = compute_mean_std(x_tr)
        x_tr, x_te = standardize(x_tr, mean, std), standardize(x_te, mean, std)

    feature_names = ["bias"] + list(labels.values())
    tx_train, tx_test = add_bias(x_tr), add_bias(x_te)

    if settings["leak_only"]:
        kept = [0] + [j for j, n in enumerate(feature_names) if n.endswith("_answered")]
        feature_names = [feature_names[j] for j in kept]
        tx_train, tx_test = tx_train[:, kept], tx_test[:, kept]

    return feature_names, tx_train, tx_test


def experiment_dir(experiment):
    return os.path.join(DATA_DIR, f"experiment_{experiment}")


def build(experiment, raw):
    settings = {**DEFAULT_SETTINGS, **EXPERIMENTS[experiment]}
    x_labels, x_train, x_test, y_train = raw
    feature_names, tx_train, tx_test = build_features(
        x_labels, x_train, x_test, y_train, settings
    )
    assert not np.isnan(tx_train).any() and not np.isnan(tx_test).any()

    output_dir = experiment_dir(experiment)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "x_train.npy"), tx_train)
    np.save(os.path.join(output_dir, "x_test.npy"), tx_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    print(f"{output_dir}: {len(feature_names)} features, tx_train {tx_train.shape}")


def pack(experiment):
    """data/experiment_X/*.npy -> data/experiment_X.npz (compressed, lossless)."""
    arrays = {
        name: np.load(os.path.join(experiment_dir(experiment), f"{name}.npy"))
        for name in ["x_train", "x_test", "y_train"]
    }
    np.savez_compressed(experiment_dir(experiment) + ".npz", **arrays)
    print(f"{experiment_dir(experiment)}.npz written")


def unpack(experiment):
    """data/experiment_X.npz -> data/experiment_X/*.npy."""
    os.makedirs(experiment_dir(experiment), exist_ok=True)
    with np.load(experiment_dir(experiment) + ".npz") as archive:
        for name in archive.files:
            np.save(
                os.path.join(experiment_dir(experiment), f"{name}.npy"), archive[name]
            )
    print(f"{experiment_dir(experiment)}/ written")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "experiments",
        nargs="+",
        help=f"experiment numbers {list(EXPERIMENTS)} or 'all'",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--unpack", action="store_true", help="extract data/experiment_X.npz"
    )
    action.add_argument(
        "--pack", action="store_true", help="compress data/experiment_X/"
    )
    args = parser.parse_args()

    experiments = (
        list(EXPERIMENTS)
        if args.experiments == ["all"]
        else [int(e) for e in args.experiments]
    )
    if args.unpack or args.pack:
        for experiment in experiments:
            unpack(experiment) if args.unpack else pack(experiment)
        return

    x_labels = load_csv_data_labels(DATA_DIR)
    x_train, x_test, y_train, _, _ = load_csv_data(DATA_DIR)
    for experiment in experiments:
        build(experiment, (x_labels, x_train, x_test, y_train))


if __name__ == "__main__":
    main()
