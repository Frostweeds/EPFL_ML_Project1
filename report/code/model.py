"""F1 score and F1-maximizing decision threshold."""

import numpy as np

N_THRESHOLDS = 100


def f1_score(y01, prediction01):
    true_positive = np.sum((prediction01 == 1) & (y01 == 1))
    false_positive = np.sum((prediction01 == 1) & (y01 == 0))
    false_negative = np.sum((prediction01 == 0) & (y01 == 1))
    return 2 * true_positive / (2 * true_positive + false_positive + false_negative)


def find_best_threshold(scores, y01):
    """Threshold maximizing F1, searched among the score quantiles."""
    thresholds = np.quantile(scores, np.linspace(0, 1, N_THRESHOLDS, endpoint=False))
    f1_per_threshold = [f1_score(y01, (scores >= t).astype(float)) for t in thresholds]
    return thresholds[int(np.argmax(f1_per_threshold))]
