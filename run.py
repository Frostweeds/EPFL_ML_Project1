"""Reproduce the submission of the report: features (features.py), logistic regression on
the features and the quadratic terms of the strongest ones (model.py), F1-maximizing
threshold.
Usage: python run.py [data_dir]   (default: ./data, ~11 minutes)
"""

import sys

import numpy as np

from features import build_features
from helpers import create_csv_submission, load_csv_data, load_csv_data_labels
from model import find_best_threshold, quadratic_logistic

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "./data"

x_labels = load_csv_data_labels(DATA_DIR)
x_train, x_test, y_train, test_ids = load_csv_data(DATA_DIR)
tx_train, tx_test = build_features(x_labels, x_train, x_test, y_train)
del x_train, x_test  # raw rows no longer needed (memory)

y01 = (y_train == 1).astype(float)

# Model selection (cross_validation.py): quadratic model
scores_train, scores_test = quadratic_logistic(y01, tx_train, tx_test)
threshold = find_best_threshold(scores_train, y01)

y_pred = np.where(scores_test >= threshold, 1, -1)
create_csv_submission(test_ids, y_pred, "submission.csv")
