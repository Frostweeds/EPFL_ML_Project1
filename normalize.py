"""Normalization of the raw BRFSS columns: special codes, then units (see codebook.py).

Every function takes `labels` first ({column index: feature name}, as returned by
helpers.load_csv_data_labels) describing the columns of `data`, and returns a copy.
"""

import numpy as np

from codebook import MISSING_CODES, NONE_CODES

# First digit = unit -> days per unit (203 = 3 per week)
FREQUENCY_UNITS = {
    "ALCDAY5": {1: 7, 2: 30},
    "FRUITJU1": {1: 1, 2: 7, 3: 30},
    "FRUIT1": {1: 1, 2: 7, 3: 30},
    "FVBEANS": {1: 1, 2: 7, 3: 30},
    "FVGREEN": {1: 1, 2: 7, 3: 30},
    "FVORANG": {1: 1, 2: 7, 3: 30},
    "VEGETAB1": {1: 1, 2: 7, 3: 30},
    "EXEROFT1": {1: 7, 2: 30},
    "EXEROFT2": {1: 7, 2: 30},
    "STRENGTH": {1: 7, 2: 30},
    "BLDSUGAR": {1: 1, 2: 7, 3: 30, 4: 365},
    "FEETCHK2": {1: 1, 2: 7, 3: 30, 4: 365},
}


def column_index(labels):
    """Name -> index."""
    return {name: i for (i, name) in labels.items()}


def replace_codes(labels, data, codes_per_column, new_value):
    data = data.copy()
    index = column_index(labels)
    for name, codes in codes_per_column.items():
        j = index[name]
        data[np.isin(data[:, j], codes), j] = new_value
    return data


def normalize_missing_codes(labels, data):
    """Don't know / refused -> NaN."""
    return replace_codes(labels, data, MISSING_CODES, np.nan)


def normalize_none_codes(labels, data):
    """None / never -> 0."""
    return replace_codes(labels, data, NONE_CODES, 0)


def normalize_weight_to_kg(labels, data):
    """WEIGHT2: pounds or 9xxx kg -> kg."""
    data = data.copy()
    j = column_index(labels)["WEIGHT2"]
    w = data[:, j]
    kg = np.full(w.shape, np.nan)
    pounds, metric = (w >= 50) & (w <= 999), (w >= 9000) & (w <= 9998)
    kg[pounds] = w[pounds] * 0.45359237
    kg[metric] = w[metric] - 9000
    data[:, j] = kg
    return data


def normalize_height_to_cm(labels, data):
    """HEIGHT3: ft/in (507) or 9xxx cm -> cm."""
    data = data.copy()
    j = column_index(labels)["HEIGHT3"]
    h = data[:, j]
    feet, inches = h // 100, h % 100
    cm = np.full(h.shape, np.nan)
    imperial = (h >= 200) & (h <= 711) & (inches < 12)
    metric = (h >= 9000) & (h <= 9998)
    cm[imperial] = (feet[imperial] * 12 + inches[imperial]) * 2.54
    cm[metric] = h[metric] - 9000
    data[:, j] = cm
    return data


def normalize_frequency_to_per_day(labels, data):
    """Frequencies -> times per day."""
    data = data.copy()
    index = column_index(labels)
    for name, days_per_unit in FREQUENCY_UNITS.items():
        j = index[name]
        v = data[:, j]
        per_day = np.full(v.shape, np.nan)
        per_day[v == 0] = 0
        for unit, days in days_per_unit.items():
            in_unit = (v // 100) == unit
            per_day[in_unit] = (v[in_unit] % 100) / days
        data[:, j] = per_day
    return data


def normalize_duration_to_days(labels, data):
    """LONGWTCH -> days."""
    data = data.copy()
    j = column_index(labels)["LONGWTCH"]
    v = data[:, j]
    days = np.full(v.shape, np.nan)
    for unit, days_per_unit in {1: 1, 2: 7, 3: 30, 4: 365}.items():
        in_unit = (v // 100) == unit
        days[in_unit] = (v[in_unit] % 100) * days_per_unit
    data[:, j] = days
    return data


def normalize_hours_minutes_to_minutes(labels, data):
    """EXERHMM1/2: 130 = 1h30 -> minutes."""
    data = data.copy()
    index = column_index(labels)
    for name in ["EXERHMM1", "EXERHMM2"]:
        v = data[:, index[name]]
        data[:, index[name]] = (v // 100) * 60 + v % 100
    return data


def normalize_month_year_to_year(labels, data):
    """FLSHTMY2, HIVTSTD3: 32015 = March 2015 -> 2015.21."""
    data = data.copy()
    index = column_index(labels)
    for name in ["FLSHTMY2", "HIVTSTD3"]:
        v = data[:, index[name]]
        month, year = v // 10000, v % 10000
        year_fraction = np.where(month == 77, 0.5, (month - 0.5) / 12)
        valid = ((month <= 12) | (month == 77)) & (year >= 1985) & (year <= 2016)
        data[:, index[name]] = np.where(valid, year + year_fraction, np.nan)
    return data


def normalize_codes(labels, data):
    """Codes first, then units."""
    data = normalize_missing_codes(labels, data)
    data = normalize_none_codes(labels, data)
    data = normalize_weight_to_kg(labels, data)
    data = normalize_height_to_cm(labels, data)
    data = normalize_frequency_to_per_day(labels, data)
    data = normalize_duration_to_days(labels, data)
    data = normalize_hours_minutes_to_minutes(labels, data)
    data = normalize_month_year_to_year(labels, data)
    return data


def normalize_nan(data, mean_per_feature=None):
    """NaN -> column mean (pass train means for x_test)."""
    if mean_per_feature is None:
        mean_per_feature = np.nanmean(data, axis=0)

    new_data = data.copy()
    row, col = np.where(np.isnan(data))
    new_data[row, col] = mean_per_feature[col]

    return new_data
