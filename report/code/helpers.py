"""Loading the course CSVs, mini-batches for SGD and writing the AIcrowd submission."""

import csv
import os

import numpy as np


def load_csv_data_labels(data_path):
    """{column index: feature name} for the columns of x_train (Id column excluded)."""
    with open(os.path.join(data_path, "x_train.csv")) as file:
        labels = file.readline().strip().split(",")
    return dict(enumerate(labels[1:]))


def load_csv_data(data_path):
    """Returns x_train, x_test, y_train in {-1, 1} and the test ids."""
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )
    return x_train[:, 1:], x_test[:, 1:], y_train, x_test[:, 0].astype(int)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a mini-batch iterator for (y, tx).

    Yields `num_batches` batches of size `batch_size` from the shuffled
    (or unshuffled) dataset. For SGD in this project, use batch_size=1.
    """
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = (batch_num * batch_size) % data_size
        end_index = min(start_index + batch_size, data_size)
        if end_index > start_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def create_csv_submission(ids, y_pred, name):
    """Columns Id and Prediction, predictions in {-1, 1}."""
    with open(name, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=["Id", "Prediction"])
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
