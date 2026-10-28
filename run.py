"""Selected model: all linear features + selected quadratics, logistic regression, lambda=1e-3.
Usage: python run.py [data_dir] [--prepared-dir data/experiment_13] [--holdout]
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
import numpy as np

from features import build_features
from helpers import create_csv_submission, load_csv_data, load_csv_data_labels
from model import f1_score
from polynomial import train_quadratic

ROOT = Path(__file__).resolve().parent


def metrics(y, scores, threshold):
    prediction, truth = scores >= threshold, y == 1
    tp, fp = int(np.sum(prediction & truth)), int(np.sum(prediction & ~truth))
    fn, tn = int(np.sum(~prediction & truth)), int(np.sum(~prediction & ~truth))
    return dict(f1=float(f1_score(y, prediction)), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), accuracy=(tp + tn) / len(y),
                log_loss=float(np.mean(np.logaddexp(0, scores) - y * scores)),
                tp=tp, fp=fp, fn=fn, tn=tn)


def load_features(data_dir, prepared_dir=None):
    """Return train/test matrices, 0/1 labels, test IDs and feature names."""
    data_dir = Path(data_dir)
    if prepared_dir is None:
        labels = load_csv_data_labels(data_dir)
        x_train, x_test, y, test_ids = load_csv_data(data_dir)
        tx, tx_test, names = build_features(labels, x_train, x_test, y, return_names=True)
    else:
        prepared_dir = Path(prepared_dir)
        tx = np.load(prepared_dir / "x_train.npy", mmap_mode="r")
        tx_test = np.load(prepared_dir / "x_test.npy", mmap_mode="r")
        y = np.load(prepared_dir / "y_train.npy")
        names = np.load(prepared_dir / "feature_names.npy")
        test_ids = np.loadtxt(data_dir / "x_test.csv", delimiter=",", skiprows=1, usecols=0, dtype=int)
    return tx, tx_test, (y == 1).astype(float), test_ids, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", type=Path, default=ROOT / "data")
    parser.add_argument("--prepared-dir", type=Path, help="Reuse the matching experiment-13 matrices")
    parser.add_argument("--holdout", action="store_true", help="Evaluate the selected model on the existing 20%% split instead of writing predictions")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    tx, tx_test, y, test_ids, names = load_features(args.data_dir, args.prepared_dir)
    result_dir = ROOT / "experiments/results/final_models"
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.holdout:
        order = np.random.default_rng(1).permutation(len(y))
        holdout, train = order[:int(0.2 * len(y))], order[int(0.2 * len(y)):]
        train_scores, holdout_scores, info = train_quadratic(y[train], tx[train], tx[holdout])
        result = dict(info, seed=1, training_rows=len(train), holdout_rows=len(holdout),
                      selected_features=names[info["selected_columns"]].tolist(),
                      train=metrics(y[train], train_scores, info["threshold"]),
                      holdout=metrics(y[holdout], holdout_scores, info["threshold"]))
        path = result_dir / "pipeline_holdout.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        m = result["holdout"]
        print(f"Holdout: F1 {m['f1']:.4f}, precision {m['precision']:.4f}, recall {m['recall']:.4f}, accuracy {m['accuracy']:.4f}, log loss {m['log_loss']:.4f}")
    else:
        assert len(tx_test) == len(test_ids), "Test matrix and identifiers must have the same rows."
        train_scores, test_scores, info = train_quadratic(y, tx, tx_test)
        predictions = np.where(test_scores >= info["threshold"], 1, -1)
        create_csv_submission(test_ids, predictions, args.output)
        result = dict(info, training_rows=len(y), test_rows=len(test_ids),
                      selected_features=names[info["selected_columns"]].tolist(),
                      train=metrics(y, train_scores, info["threshold"]), submission=str(args.output))
        (result_dir / "final_training.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {len(predictions)} predictions to {args.output}")
    print(f"{info['columns']} columns; final gradient {info['gradient']:.2e} ({info['steps']} steps)")


if __name__ == "__main__":
    main()
