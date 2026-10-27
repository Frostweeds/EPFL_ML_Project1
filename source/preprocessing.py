"""Feature processing building blocks: missing indicators, one-hot expansion, column
filters, outlier clipping, standardization.

`labels` is always {column index: feature name} describing the columns of `data`.
"""

import numpy as np

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
    "EYEEXAM",  # time since eye exam, 8 = never
    "EXRACT11",  # type of physical activity
    "EXRACT21",
    "IMFVPLAC",  # place of the flu shot
    "CRGVREL1",  # relationship to the care recipient
    "CRGVPRB1",  # main health problem of the care recipient
    "TETANUS",  # yes Tdap / yes not Tdap / yes unknown / no
    "ASPUNSAF",  # aspirin unsafe: yes not stomach / yes stomach / no
    "SCNTPAID",  # salary / hourly / by the job / other
    "SCNTLPAD",
    "_CPRACE",  # child race
    "_RACE",  # race / ethnicity
]

# Only asked after a heart attack, part of the target: a target leak
LEAK_COLUMNS = ["HAREHAB1_missing"]

# Missing indicator: NaN ratio above, and at least that many NaN rows
MISSING_INDICATOR_THRESHOLD = 0.05
MIN_ROWS = 100

# 0.95 would drop real conditions (CVDSTRK3)
CONSTANT_THRESHOLD = 0.99

# NaN ratio above which a column is dropped
INCOMPLETE_THRESHOLD = 0.99

# |correlation with the output| below which a column is dropped
OUTPUT_CORRELATION_THRESHOLD = 0.01

# 0.8 would drop distinct features (_BMI5)
CORRELATION_THRESHOLD = 0.95

# Clip to the [ratio, 1 - ratio] quantiles
OUTLIER_RATIO = 0.01

# Fewer distinct values = categorical, never clipped
MIN_CONTINUOUS_VALUES = 10


def find_missing_indicator_columns(labels, data):
    nan_count = np.isnan(data).sum(axis=0)
    selected = (nan_count / data.shape[0] > MISSING_INDICATOR_THRESHOLD) & (
        nan_count >= MIN_ROWS
    )
    return [labels[j] for j in np.flatnonzero(selected)]


def add_missing_indicator_columns(labels, data, filter_names):
    """For each column in filter_names, append a 0/1 column `column_missing` (1 = NaN)."""
    index = {name: i for (i, name) in labels.items()}
    indicators = np.isnan(data[:, [index[name] for name in filter_names]]).astype(float)
    new_names = list(labels.values()) + [f"{name}_missing" for name in filter_names]
    return dict(enumerate(new_names)), np.c_[data, indicators]


def expand_category_column(labels, data, column, categories):
    """One-hot: `column` -> one 0/1 column `column_X` per category X, in place of `column`
    (NaN -> all 0). Returns the new labels and data."""
    j = {name: i for (i, name) in labels.items()}[column]
    one_hot = (data[:, j, None] == categories[None, :]).astype(float)
    names = list(labels.values())
    new_names = names[:j] + [f"{column}_{c:g}" for c in categories] + names[j + 1 :]
    return dict(enumerate(new_names)), np.c_[data[:, :j], one_hot, data[:, j + 1 :]]


def output_correlation(column, y):
    """|Correlation| feature - output, NaN rows ignored."""
    m = ~np.isnan(column)
    return abs(np.corrcoef(column[m], y[m])[0, 1])


def find_constant_columns(labels, data):
    """Most frequent value in >= CONSTANT_THRESHOLD of the non-NaN rows. Missing
    indicators are rare by design and never constant."""
    constant = []
    for j, column in enumerate(data.T):
        values = column[~np.isnan(column)]
        top_ratio = np.unique(values, return_counts=True)[1].max() / values.size
        if top_ratio >= CONSTANT_THRESHOLD and not labels[j].endswith("_missing"):
            constant.append(labels[j])
    return constant


def find_incomplete_columns(labels, data):
    nan_ratio_per_feature = np.isnan(data).mean(axis=0)
    return [
        labels[j] for j in np.flatnonzero(nan_ratio_per_feature > INCOMPLETE_THRESHOLD)
    ]


def find_low_output_correlation_columns(labels, data, y):
    correlations = np.array([output_correlation(column, y) for column in data.T])
    return [
        labels[j] for j in np.flatnonzero(correlations < OUTPUT_CORRELATION_THRESHOLD)
    ]


def find_correlated_columns(labels, data, y):
    """Feature - feature, after mean imputation: per pair with |correlation| >=
    CORRELATION_THRESHOLD, the one less correlated with the output."""
    feature_correlation = np.abs(
        np.corrcoef(normalize_nan(data, np.nanmean(data, axis=0)), rowvar=False)
    )
    output_correlations = np.array([output_correlation(column, y) for column in data.T])

    # Pairs i < j, most correlated first
    pair_i, pair_j = np.triu_indices(data.shape[1], k=1)
    pair_correlation = feature_correlation[pair_i, pair_j]
    correlated_indices = set()
    for k in np.argsort(pair_correlation)[::-1]:
        if pair_correlation[k] < CORRELATION_THRESHOLD:
            break
        i, j = pair_i[k], pair_j[k]
        if i in correlated_indices or j in correlated_indices:
            continue
        if output_correlations[i] < output_correlations[j]:
            correlated_indices.add(i)
        else:
            correlated_indices.add(j)
    return [labels[j] for j in sorted(correlated_indices)]


def fit_outlier_bounds(data):
    """[OUTLIER_RATIO, 1 - OUTLIER_RATIO] quantiles of the continuous columns, infinite
    bounds for the categorical ones."""
    lower = np.nanquantile(data, OUTLIER_RATIO, axis=0)
    upper = np.nanquantile(data, 1 - OUTLIER_RATIO, axis=0)
    distinct_values = np.array(
        [np.unique(column[~np.isnan(column)]).size for column in data.T]
    )
    categorical = distinct_values <= MIN_CONTINUOUS_VALUES
    lower[categorical], upper[categorical] = -np.inf, np.inf
    return lower, upper


def standardize(data, mean, std):
    return (data - mean) / std


def add_bias(data):
    return np.c_[np.ones(data.shape[0]), data]
