"""Reproduce the submission of the report: features (features.py), logistic regression
(implementations.py), F1-maximizing threshold. Run from this folder: python run.py (~2 minutes).
"""

import numpy as np

from features import build_features
from helpers import create_csv_submission, load_csv_data, load_csv_data_labels
from implementations import logistic_regression
from model import find_best_threshold

DATA_DIR = "../../data"

# Model selection (experiments/): lambda = 0, gradient descent converged at 300 steps
MAX_ITERS = 300
GAMMA = 1.0

x_labels = load_csv_data_labels(DATA_DIR)
x_train, x_test, y_train, test_ids = load_csv_data(DATA_DIR)
tx_train, tx_test = build_features(x_labels, x_train, x_test, y_train)

y01 = (y_train == 1).astype(float)
initial_w = np.zeros(tx_train.shape[1])
w, _ = logistic_regression(y01, tx_train, initial_w, MAX_ITERS, GAMMA)
threshold = find_best_threshold(tx_train.dot(w), y01)

y_pred = np.where(tx_test.dot(w) >= threshold, 1, -1)
create_csv_submission(test_ids, y_pred, "submission.csv")
