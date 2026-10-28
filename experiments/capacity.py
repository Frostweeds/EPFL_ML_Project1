"""Does more model capacity help? Logistic vs interactions vs MLP vs k-NN, same 3-fold CV.

Usage: python experiments/capacity.py [dataset_dir]   (default: data/experiment_5)
Every model returns scores; the threshold maximizing F1 is chosen on training scores,
then applied to the held-out fold (same protocol as baselines_cv.py).
Writes experiments/results/capacity_<dataset>.json.
"""

import json
import os
import sys
import time
from itertools import combinations_with_replacement

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baselines_cv import K, ROOT, SEED, best_threshold, f1, impl, lambda_max, make_folds, split  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "data/experiment_5"


# ---------------------------------------------------------------------------- models
# Each model: (tx_tr (N, D), y01_tr (N,), tx_te (M, D)) -> (scores_tr (N,), scores_te (M,))


def logistic_gd(tx, y01):
    """Logistic regression, safe GD step. In: tx (N, D), y01 (N,). Out: w (D,)."""
    L = lambda_max(tx) / len(y01)
    gamma = min(1.0, 1.9 / (L / 4))
    w, _ = impl.logistic_regression(y01, tx, np.zeros(tx.shape[1]), int(np.ceil(300 / gamma)), gamma)
    return w


def logistic(tx_tr, y01_tr, tx_te):
    """Baseline: logistic regression, gamma = 1, 300 steps (as in the baselines)."""
    w, _ = impl.logistic_regression(y01_tr, tx_tr, np.zeros(tx_tr.shape[1]), 300, 1.0)
    return tx_tr @ w, tx_te @ w


def top_features(tx_tr, y01_tr, k):
    """Indices of the k features with largest |w| in a logistic fit (bias excluded). Out: (k,)."""
    w, _ = impl.logistic_regression(y01_tr, tx_tr, np.zeros(tx_tr.shape[1]), 300, 1.0)
    return np.argsort(np.abs(w[1:]))[-k:] + 1


def interactions(k):
    """Logistic + all degree-2 products of the top-k features (chosen on the training folds)."""

    def model(tx_tr, y01_tr, tx_te):
        top = top_features(tx_tr, y01_tr, k)
        pairs = list(combinations_with_replacement(top, 2))  # k (k + 1) / 2 products
        p_tr = np.column_stack([tx_tr[:, i] * tx_tr[:, j] for i, j in pairs])
        p_te = np.column_stack([tx_te[:, i] * tx_te[:, j] for i, j in pairs])
        mean, std = p_tr.mean(axis=0), p_tr.std(axis=0) + 1e-12  # standardize on training folds
        x_tr = np.hstack([tx_tr, (p_tr - mean) / std])
        x_te = np.hstack([tx_te, (p_te - mean) / std])
        w = logistic_gd(x_tr, y01_tr)
        return x_tr @ w, x_te @ w

    return model


def mlp(hidden, epochs, lr=1e-3, batch=512, weight_decay=1e-5):
    """One-hidden-layer ReLU network, logistic loss, Adam (numpy only)."""

    def model(tx_tr, y01_tr, tx_te):
        rng = np.random.default_rng(SEED)
        n, d = tx_tr.shape
        params = {
            "W1": rng.normal(0, np.sqrt(2 / d), (d, hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0, np.sqrt(1 / hidden), hidden),
            "b2": np.array([np.log(y01_tr.mean() / (1 - y01_tr.mean()))]),  # start at the base rate
        }
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        step = 0
        for _ in range(epochs):
            order = rng.permutation(n)  # one shuffle per epoch
            for start in range(0, n, batch):
                idx = order[start:start + batch]
                x, y = tx_tr[idx], y01_tr[idx]
                h = np.maximum(x @ params["W1"] + params["b1"], 0)  # (B, H)
                z = h @ params["w2"] + params["b2"]  # (B,)
                g = (1 / (1 + np.exp(-z)) - y) / len(y)  # d loss / d z
                dh = np.outer(g, params["w2"]) * (h > 0)  # backprop through ReLU
                grads = {"W1": x.T @ dh, "b1": dh.sum(0), "w2": h.T @ g, "b2": np.array([g.sum()])}
                grads["W1"] += weight_decay * params["W1"]
                step += 1
                for key in params:  # Adam
                    m[key] = 0.9 * m[key] + 0.1 * grads[key]
                    v[key] = 0.999 * v[key] + 0.001 * grads[key] ** 2
                    m_hat, v_hat = m[key] / (1 - 0.9**step), v[key] / (1 - 0.999**step)
                    params[key] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        def scores(x):
            return np.maximum(x @ params["W1"] + params["b1"], 0) @ params["w2"] + params["b2"]

        return scores(tx_tr), scores(tx_te)

    return model


def knn(k, n_features=20, n_ref=50_000, n_eval=10_000):
    """k-NN on the top features, subsampled. Score = fraction of positive neighbors."""

    def neighbors_score(ref, y_ref, x):
        out = np.empty(len(x))
        sq_ref = (ref**2).sum(1)
        for s in range(0, len(x), 1000):  # chunks to bound memory
            c = x[s:s + 1000]
            dist = (c**2).sum(1)[:, None] + sq_ref[None, :] - 2 * c @ ref.T
            nn = np.argpartition(dist, k, axis=1)[:, :k]
            out[s:s + 1000] = y_ref[nn].mean(1)
        return out

    def model(tx_tr, y01_tr, tx_te):
        rng = np.random.default_rng(SEED)
        top = top_features(tx_tr, y01_tr, n_features)
        perm = rng.permutation(len(y01_tr))
        ref, thr = perm[:n_ref], perm[n_ref:n_ref + n_eval]  # disjoint: no self-neighbor
        x_tr = tx_tr[:, top]
        scores_thr = neighbors_score(x_tr[ref], y01_tr[ref], x_tr[thr])
        scores_te = neighbors_score(x_tr[ref], y01_tr[ref], tx_te[:, top])
        return scores_thr, scores_te, thr  # threshold chosen on the thr subset

    return model


# ---------------------------------------------------------------------------- protocol


def cross_validate_scores(model, tx, y01, n_test=None):
    """K-fold CV of a scoring model. Out: list of K dicts (held-out F1, train F1, seconds)."""
    folds = make_folds(len(y01))
    rng = np.random.default_rng(SEED)
    results = []
    for k in range(K):
        te = folds[k]
        tr = np.concatenate([folds[i] for i in range(K) if i != k])
        if n_test is not None:  # k-NN only: evaluate on a random subset of the held-out fold
            te = rng.choice(te, n_test, replace=False)
        start = time.time()
        out = model(tx[tr], y01[tr], tx[te])
        if len(out) == 3:  # k-NN: training scores are on a subset of tr
            s_tr, s_te, sub = out
            y_tr = y01[tr][sub]
        else:
            s_tr, s_te = out
            y_tr = y01[tr]
        t = best_threshold(s_tr, y_tr)  # threshold chosen on training scores only
        results.append(
            {"f1": f1(y01[te], s_te >= t), "f1_train": f1(y_tr, s_tr >= t), "seconds": time.time() - start}
        )
    return results


def main():
    tx, y = np.load(f"{DATASET}/x_train.npy"), np.load(f"{DATASET}/y_train.npy")
    tr, _ = split(len(y))  # the 20% hold-out split is not touched here
    tx, y01 = tx[tr], (y[tr] == 1).astype(float)
    print(f"{DATASET}: tx {tx.shape}", flush=True)

    models = {
        "logistic (reference)": (logistic, None),
        "logistic + interactions top-10 (55)": (interactions(10), None),
        "logistic + interactions top-20 (210)": (interactions(20), None),
        "logistic + interactions top-30 (465)": (interactions(30), None),
        "MLP 64 hidden, 5 epochs": (mlp(64, 5), None),
        "MLP 256 hidden, 10 epochs": (mlp(256, 10), None),
        "k-NN k=25, top-20 features": (knn(25), 10_000),
        "k-NN k=101, top-20 features": (knn(101), 10_000),
    }
    results = {}
    for name, (model, n_test) in models.items():
        folds = cross_validate_scores(model, tx, y01, n_test)
        results[name] = folds
        f = [r["f1"] for r in folds]
        print(
            f"{name:<40} F1 {np.mean(f):.4f} ± {np.std(f):.4f} | train {np.mean([r['f1_train'] for r in folds]):.4f}"
            f" | per fold {' / '.join(f'{x:.4f}' for x in f)} | {sum(r['seconds'] for r in folds):.0f}s",
            flush=True,
        )

    path = f"{ROOT}/experiments/results/capacity_{os.path.basename(DATASET.rstrip('/'))}.json"
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"written {path}")


if __name__ == "__main__":
    main()
