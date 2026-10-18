"""Feature processing building blocks: drop columns, find constant / incomplete / correlated
columns, clip outliers, standardize.

`labels` is always {column index: feature name} describing the columns of `data`.
Fit statistics (columns to drop, outlier bounds, mean, std) on x_train only and reuse them for x_test.
"""

import numpy as np

from implementations import build_poly
from normalize import normalize_nan

# Dates, IDs, phone screening
IDENTIFIER_COLUMNS = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "_STSTR",
    "CTELENUM",
    "CTELNUM1",
    "STATERES",
    "CSTATE",
]

# Survey weights / sampling design: how the respondent was sampled, not their health
SURVEY_DESIGN_COLUMNS = [
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_DUALUSE",
    "_DUALCOR",
    "_LLCPWT",
    "QSTVER",
]

# Nominal columns (arbitrary codes, or an ordered scale with an off-scale code such as
# 8 = never / not applicable), one-hot expanded. Ordinal scales (GENHLTH, INCOME2, ...) stay numeric.
CATEGORY_COLUMNS = [
    "PERSDOC2",  # one / more than one / no personal doctor
    "CHECKUP1",  # time since checkup, 8 = never
    "DIABETE3",  # yes / during pregnancy / no / prediabetes
    "PREDIAB1",  # yes / during pregnancy / no
    "MARITAL",
    "RENTHOM1",  # own / rent / other
    "EMPLOY1",
    "LASTSMK2",  # time since quitting, 8 = never smoked regularly
    "SEATBELT",  # always ... never, 8 = never drive
    "WHRTST10",  # place of last HIV test
    "SCNTMNY1",  # always ... never worried about money, 8 = not applicable
    "SCNTMEL1",  # same for meals
    "RCSRLTN2",  # relationship to the child
    "MSCODE",  # metropolitan status
    "_CHOLCHK",  # checked within 5 years / never / more than 5 years ago
    "_CRACE1",
    "_PRACE1",
    "_MRACE1",
    "_RACEGR3",
    "_RACE_G1",
    "_PAREC1",  # aerobic and strength guidelines met: both / aerobic / strength / neither
    "_LMTACT1",  # arthritis: limited / not limited / no arthritis
    "_LMTWRK1",
    "_LMTSCL1",
]

# 0.95 would drop real conditions (CVDSTRK3)
CONSTANT_THRESHOLD = 0.99

# 0.8 would drop distinct features (_BMI5)
CORRELATION_THRESHOLD = 0.95

# 0.01 drops 17 / 161 columns left after the previous filters
OUTPUT_CORRELATION_THRESHOLD = 0.01

MIN_ROWS = 100  # fewer rows = noise

# NaN ratio above which a column is dropped (0.9 drops 97 / 292 columns)
INCOMPLETE_THRESHOLD = 0.9

# NaN ratio above which a column gets a `column_missing` indicator
MISSING_INDICATOR_THRESHOLD = 0.05

# Clip to the [ratio, 1 - ratio] quantiles
OUTLIER_RATIO = 0.01

# Fewer distinct values = categorical, never clipped
MIN_CONTINUOUS_VALUES = 10


def drop_labels(labels, filter_names):
    """Index -> name after dropping `names`."""
    return dict(enumerate(name for name in labels.values() if name not in filter_names))


def drop_columns_by_name(labels, data, filter_names):
    assert data.shape[1] == len(labels), "labels must describe the columns of data"
    kept_indices = [i for (i, name) in labels.items() if name not in filter_names]
    return data[:, kept_indices]


# Above this many distinct values, a column is not expanded
MAX_CATEGORIES = 10


def expand_category_column(labels, data, column, categories=None, nan_category=False):
    """One-hot: `column` -> one 0/1 column `column_X` per category X, in place of `column`.

    NaN -> all 0, or its own `column_nan` category with nan_category=True. `categories`
    defaults to the distinct values of the column: pass the ones returned for x_train when
    expanding x_test so both get the same columns (a NaN in them = NaN category).
    Returns the new labels, the new data and the categories used.
    """
    assert data.shape[1] == len(labels), "labels must describe the columns of data"
    j = {name: i for (i, name) in labels.items()}[column]
    values = data[:, j]
    if categories is None:
        categories = np.unique(values[~np.isnan(values)])
        if nan_category:
            categories = np.append(categories, np.nan)
    categories = np.asarray(categories, dtype=float)
    assert (
        len(categories) < MAX_CATEGORIES
    ), f"{column} has {len(categories)} distinct values, expected < {MAX_CATEGORIES}"

    one_hot = (values[:, None] == categories[None, :]).astype(float)
    one_hot[:, np.isnan(categories)] = np.isnan(values)[:, None]
    names = list(labels.values())
    new_names = (
        names[:j]
        + [f"{column}_nan" if np.isnan(c) else f"{column}_{c:g}" for c in categories]
        + names[j + 1 :]
    )
    new_data = np.c_[data[:, :j], one_hot, data[:, j + 1 :]]
    return dict(enumerate(new_names)), new_data, categories


def top_value_ratio(column):
    """Share of the most frequent value (NaN ignored)."""
    values = column[~np.isnan(column)]
    if not values.size:
        return np.nan
    return np.unique(values, return_counts=True)[1].max() / values.size


def output_correlation(column, y):
    """|Correlation| feature - output, NaN rows ignored."""
    m = ~np.isnan(column)
    if m.sum() < MIN_ROWS or np.std(column[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return abs(np.corrcoef(column[m], y[m])[0, 1])


def find_constant_columns(labels, data, threshold=CONSTANT_THRESHOLD):
    ratio_per_feature = np.array([top_value_ratio(column) for column in data.T])
    return [labels[j] for j in np.flatnonzero(ratio_per_feature >= threshold)]


def add_polynomial_columns(labels, data, filter_names, degree=2):
    """For each column in filter_names, append its powers 2..degree as `column_powD` columns.

    Apply after NaN filling and standardization: NaN stay NaN, and powers of centered values
    are far less correlated with the column itself. Returns the new labels and data.
    """
    assert data.shape[1] == len(labels), "labels must describe the columns of data"
    index = {name: i for (i, name) in labels.items()}
    # build_poly gives [1, x, x^2, ...]: keep the powers >= 2
    powers = [build_poly(data[:, index[name]], degree)[:, 2:] for name in filter_names]
    new_names = list(labels.values()) + [
        f"{name}_pow{d}" for name in filter_names for d in range(2, degree + 1)
    ]
    return dict(enumerate(new_names)), np.c_[(data, *powers)]


def add_missing_indicator_columns(labels, data, filter_names):
    """For each column in filter_names, append a 0/1 column `column_missing` (1 = NaN).

    Returns the new labels and data.
    """
    assert data.shape[1] == len(labels), "labels must describe the columns of data"
    index = {name: i for (i, name) in labels.items()}
    indicators = np.isnan(data[:, [index[name] for name in filter_names]]).astype(float)
    new_names = list(labels.values()) + [f"{name}_missing" for name in filter_names]
    return dict(enumerate(new_names)), np.c_[data, indicators]


def filter_incomplete_column(labels, data, nan_ratio=INCOMPLETE_THRESHOLD):
    """Columns with a NaN ratio > nan_ratio."""
    nan_ratio_per_feature = np.isnan(data).mean(axis=0)
    return [labels[j] for j in np.flatnonzero(nan_ratio_per_feature > nan_ratio)]


def find_correlated_columns(labels, data, y, threshold=CORRELATION_THRESHOLD):
    """Feature - feature: per pair with |correlation| >= threshold, the one less correlated with the output."""
    with np.errstate(divide="ignore", invalid="ignore"):
        feature_correlation = np.abs(np.corrcoef(normalize_nan(data), rowvar=False))

    # Pairs i < j
    pair_i, pair_j = np.triu_indices(data.shape[1], k=1)
    pair_correlation = feature_correlation[pair_i, pair_j]

    output_correlation_or_zero = np.nan_to_num(
        np.array([output_correlation(column, y) for column in data.T])
    )

    correlated_indices = set()
    for k in np.argsort(np.nan_to_num(pair_correlation))[::-1]:
        if not pair_correlation[k] >= threshold:
            break
        i, j = pair_i[k], pair_j[k]
        if i in correlated_indices or j in correlated_indices:
            continue
        if output_correlation_or_zero[i] < output_correlation_or_zero[j]:
            correlated_indices.add(i)
        else:
            correlated_indices.add(j)

    return [labels[j] for j in sorted(correlated_indices)]


def find_low_output_correlation_columns(
    labels, data, y, threshold=OUTPUT_CORRELATION_THRESHOLD
):
    """Feature - output: |correlation| < threshold (not computable counts as 0)."""
    output_correlation_or_zero = np.nan_to_num(
        np.array([output_correlation(column, y) for column in data.T])
    )
    return [labels[j] for j in np.flatnonzero(output_correlation_or_zero < threshold)]


def filter_outliers(data, ratio=OUTLIER_RATIO, bounds=None):
    """Clip continuous columns to their [ratio, 1 - ratio] quantiles (NaN kept).

    Returns the clipped data and the (lower, upper) bounds: pass the train bounds for x_test.
    """
    if bounds is None:
        lower = np.nanquantile(data, ratio, axis=0)
        upper = np.nanquantile(data, 1 - ratio, axis=0)
        distinct_values = np.array(
            [np.unique(column[~np.isnan(column)]).size for column in data.T]
        )
        categorical = distinct_values <= MIN_CONTINUOUS_VALUES
        lower[categorical], upper[categorical] = -np.inf, np.inf
        bounds = (lower, upper)

    return np.clip(data, *bounds), bounds


def compute_mean_std(data):
    """Column mean and std (NaN ignored), std = 0 -> 1."""
    mean = np.nan_to_num(np.nanmean(data, axis=0), nan=0.0)
    std = np.nan_to_num(np.nanstd(data, axis=0), nan=1.0)
    std[std == 0] = 1.0
    return mean, std


def standardize(data, mean, std):
    return (data - mean) / std


def add_bias(data):
    return np.c_[np.ones(data.shape[0]), data]
