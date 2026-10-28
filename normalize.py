"""Normalization of the raw BRFSS columns: special codes, then units.

Special value codes per column are taken from the BRFSS 2015 codebook:
https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
(linked from https://www.cdc.gov/brfss/annual_data/annual_2015.html).
"""

import numpy as np

# "Don't know / Not sure", "Refused" and "Missing" answers, to be replaced by NaN.
# Codes -> columns, in codebook order within each group.
# fmt: off
MISSING_CODES_COLUMNS = {
    (7, 9): """
        LANDLINE GENHLTH HLTHPLN1 PERSDOC2 MEDCOST CHECKUP1 BPHIGH4 BPMEDS BLOODCHO
        CHOLCHK TOLDHI2 CVDSTRK3 ASTHMA3 ASTHNOW CHCSCNCR CHCOCNCR CHCCOPD1 HAVARTH3
        ADDEPEV2 CHCKIDNY DIABETE3 RENTHOM1 NUMHHOL2 NUMPHON2 CPDEMO1 VETERAN3 INTERNET
        PREGNANT QLACTLM2 USEEQUIP BLIND DECIDE DIFFWALK DIFFDRES DIFFALON SMOKE100
        SMOKDAY2 STOPSMK2 USENOW3 EXERANY2 LMTJOIN3 ARTHDIS2 ARTHSOCL SEATBELT FLUSHOT6
        PNEUVAC3 HIVTST6 PDIABTST PREDIAB1 EYEEXAM DIABEYE DIABEDU CAREGIV1 CRGVLNG1
        CRGVHRS1 CRGVPERS CRGVHOUS CRGVMST2 CRGVEXPT VIPRFVS2 VIEYEXM2 VIINSUR2 CIMEMLOS
        CDHOUSE CDASSIST CDHELP CDSOCIAL CDDISCUS WTCHSALT DRADVISE ASYMPTOM ASTHMED3
        ASINHALR HAREHAB1 STREHAB1 CVDASPRN ASPUNSAF RLIVPAIN RDUCHART RDUCSTRK ARTTODAY
        ARTHWGT ARTHEXER ARTHEDU TETANUS HPVADVC2 SHINGLE2 HADMAM HOWLONG HADPAP2
        LASTPAP2 HPVTEST HPLSTTST HADHYST2 PROFEXAM LENGEXAM BLDSTOOL LSTBLDS3 HADSIGM3
        HADSGCO1 LASTSIG3 PCPSAAD2 PCPSADI1 PCPSARE1 PSATEST1 PSATIME PCPSARS1 PCDMDECN
        SCNTMNY1 SCNTMEL1 SCNTPAID SCNTLPAD SXORIENT TRNSGNDR RCSRLTN2 CASTHDX2 CASTHNO2
        EMTSUPRT LSATISFY MISTMNT ADANXEV DRNKANY5
    """.split(),
    (77, 99): """
        HHADULT PHYSHLTH MENTHLTH POORHLTH INCOME2 LASTSMK2 AVEDRNK2 DRNK3GE5 MAXDRNKS
        EXRACT11 EXRACT21 JOINPAIN IMFVPLAC WHRTST10 DOCTDIAB CHKHEMO3 FEETCHK CRGVREL1
        CRGVPRB1 VINOCRE2 HPVADSHT ADPLEASR ADDOWN ADSLEEP ADENERGY ADEAT1 ADFAIL
        ADTHINK ADMOVE _CRACE1 _CPRACE _PRACE1 _MRACE1
    """.split(),
    (777, 999): """
        ALCDAY5 FRUITJU1 FRUIT1 FVBEANS FVGREEN FVORANG VEGETAB1 EXEROFT1 EXERHMM1
        EXEROFT2 EXERHMM2 STRENGTH BLDSUGAR FEETCHK2 LONGWTCH ASACTLIM
    """.split(),
    (7777, 9999): ["WEIGHT2", "HEIGHT3"],
    (777777, 999999): ["FLSHTMY2", "HIVTSTD3"],
    (9,): """
        MARITAL EDUCA EMPLOY1 INSULIN PCPSADE1 RCSGENDR _CHISPNC _RFHLTH _HCVU651
        _RFHYPE5 _CHOLCHK _RFCHOL _LTASTH1 _CASTHM1 _ASTHMS1 _HISPANC _RACE _RACEG21
        _RACEGR3 _RFBMI5 _CHLDCNT _EDUCAG _INCOMG _SMOKER3 _RFSMOK3 _RFBING5 _RFDRHV5
        _FRTLT1 _VEGLT1 _TOTINDA PAMISS1_ _PACAT1 _PAINDX1 _PA150R2 _PA300R2 _PA30021
        _PASTRNG _PAREC1 _PASTAE1 _LMTACT1 _LMTWRK1 _LMTSCL1 _RFSEAT2 _RFSEAT3 _FLSHOT6
        _PNEUMO2 _AIDTST3
    """.split(),
    (99,): ["CHILDREN", "PAFREQ1_", "PAFREQ2_", "STRFREQ_"],
    (999,): ["MAXVO2_", "FC60_"],
    (7,): "VIDFCLT2 VIREDIF3 VICTRCT4 VIGLUMA2 VIMACDG2 ASATTACK ASNOSLEP".split(),
    (98, 99): ["DIABAGE2", "ASTHMAGE", "ASRCHKUP"],
    (98,): ["ASERVIST", "ASDRVIST"],
    (97, 99): ["SCNTWRK1", "SCNTLWK1"],
    (14,): ["_AGEG5YR"],
    (3,): ["_AGE65YR"],
    (99999,): ["WTKG3"],
    (900,): ["DROCDY3_"],
    (99900,): ["_DRNKWEK"],
}

# "None" / "Never" answers of count or frequency questions (e.g. PHYSHLTH 88 = 0 days), to
# be replaced by 0. Codes -> columns.
NONE_CODES_COLUMNS = {
    (88,): """
        PHYSHLTH MENTHLTH POORHLTH CHILDREN DRNK3GE5 DOCTDIAB FEETCHK ASERVIST ASDRVIST
        ASRCHKUP ADPLEASR ADDOWN ADSLEEP ADENERGY ADEAT1 ADFAIL ADTHINK ADMOVE
    """.split(),
    (888,): ["ALCDAY5", "STRENGTH", "BLDSUGAR", "FEETCHK2", "ASACTLIM"],
    (555,): ["FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1"],
    (88, 98): ["CHKHEMO3"],  # 98 = never heard of the A1C test
    (98,): ["SCNTWRK1", "SCNTLWK1"],  # 98 = zero hours (does not work)
}
# fmt: on

# Column -> codes, as used by replace_codes
MISSING_CODES = {
    name: list(codes)
    for codes, names in MISSING_CODES_COLUMNS.items()
    for name in names
}

NONE_CODES = {
    name: list(codes) for codes, names in NONE_CODES_COLUMNS.items() for name in names
}

# First digit = unit, then the count (203 = 3 per week). Unit -> days per unit
PER_WEEK_MONTH = {1: 7, 2: 30}
PER_DAY_WEEK_MONTH = {1: 1, 2: 7, 3: 30}
PER_DAY_WEEK_MONTH_YEAR = {1: 1, 2: 7, 3: 30, 4: 365}

FREQUENCY_UNITS = {
    "ALCDAY5": PER_WEEK_MONTH,
    "FRUITJU1": PER_DAY_WEEK_MONTH,
    "FRUIT1": PER_DAY_WEEK_MONTH,
    "FVBEANS": PER_DAY_WEEK_MONTH,
    "FVGREEN": PER_DAY_WEEK_MONTH,
    "FVORANG": PER_DAY_WEEK_MONTH,
    "VEGETAB1": PER_DAY_WEEK_MONTH,
    "EXEROFT1": PER_WEEK_MONTH,
    "EXEROFT2": PER_WEEK_MONTH,
    "STRENGTH": PER_WEEK_MONTH,
    "BLDSUGAR": PER_DAY_WEEK_MONTH_YEAR,
    "FEETCHK2": PER_DAY_WEEK_MONTH_YEAR,
}


def column_index(labels):
    """Name -> index."""
    return {name: i for (i, name) in labels.items()}


def replace_codes(labels, data, codes_per_column, new_value):
    """In each column of codes_per_column, its codes -> new_value. Returns a copy."""
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
    for unit, days_per_unit in PER_DAY_WEEK_MONTH_YEAR.items():
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


def normalize_nan(data, mean_per_feature):
    """NaN -> column mean (pass the train means for x_test)."""
    new_data = data.copy()
    row, col = np.where(np.isnan(data))
    new_data[row, col] = mean_per_feature[col]

    return new_data
