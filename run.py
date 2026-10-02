"""Reproducibility script for EPFL ML Project 1.

Running `python run.py` must regenerate exactly the .csv submission used for
the team's best AIcrowd score (see the project statement, Step 4).

This is a thin script: it should just load data, apply the chosen
preprocessing/feature engineering, train the final model using functions
from implementations.py, and write the submission file. Keep the actual
exploration/model-selection work in notebooks/project1.ipynb; only the
final, decided pipeline belongs here.
"""

import numpy as np

from helpers import load_csv_data, create_csv_submission
import implementations as impl

DATA_DIR = "data"
OUTPUT_PATH = "submission.csv"


def build_features(x):
    """Apply the final feature engineering / preprocessing pipeline.

    Must be applied identically to x_train and x_test.
    """
    # TODO: cleaning, scaling, feature construction, add bias column, etc.
    raise NotImplementedError


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_DIR)

    tx_train = build_features(x_train)
    tx_test = build_features(x_test)

    # TODO: replace with the final chosen method + hyperparameters
    # e.g. w, loss = impl.ridge_regression(y_train, tx_train, lambda_=...)
    w, loss = None, None
    raise NotImplementedError("Fill in the final training call above.")

    y_pred = tx_test @ w
    y_pred = np.where(y_pred >= 0, 1, -1)

    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)


if __name__ == "__main__":
    main()
