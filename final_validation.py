"""Final three-fold polynomial comparison and existing 20% holdout evaluation.

Run --fold 1, --fold 2 and --fold 3 independently, then --holdout and --collect.
With no action flag, run all folds, both holdout fits and collection.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
import numpy as np

from features import add_harehab1
from helpers import load_harehab1
from implementations import logistic_regression
from model import find_best_threshold
from polynomial import (
    GAMMA, MAX_ITERS, WEIGHT_FRACTION, append_cubic, fit_polynomial, fit_quadratic,
    quadratic_features, select_features, train_quadratic,
)
from run import ROOT, load_features, metrics

SEED = 1
K_FOLD = 3
HOLDOUT_RATIO = 0.2
LAMBDAS = (0.0, 1e-3)
TOP = 10
OUTPUT_DIR = ROOT / "experiments/results/final_models"


def save(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def splits(y):
    order = np.random.default_rng(SEED).permutation(len(y))
    cut = int(HOLDOUT_RATIO * len(y))
    holdout, train = order[:cut], order[cut:]
    inner = np.random.default_rng(SEED).permutation(len(train))
    folds = [train[rows] for rows in np.array_split(inner, K_FOLD)]
    return train, holdout, folds


def scored_record(y_train, y_valid, train_scores, valid_scores, info, split="validation"):
    threshold = info["threshold"]
    return dict(
        info, training_rows=len(y_train), evaluation_rows=len(y_valid),
        train=metrics(y_train, train_scores, threshold),
        **{split: metrics(y_valid, valid_scores, threshold)},
        evaluation_f1_t0=metrics(y_valid, valid_scores, 0)["f1"],
    )


def fit_evaluate(tx, y, names, train, valid, split="validation"):
    train_scores, valid_scores, info = train_quadratic(y[train], tx[train], tx[valid])
    info.update(selected_features=names[info["selected_columns"]].tolist(),
                degree=2, weight_fraction=WEIGHT_FRACTION)
    return scored_record(y[train], y[valid], train_scores, valid_scores, info, split)


def progress(row, label):
    split = "validation" if "validation" in row else "holdout"
    print(f"  {label}: train F1 {row['train']['f1']:.4f}, {split} F1 "
          f"{row[split]['f1']:.4f}, F1(t=0) {row['evaluation_f1_t0']:.4f}; "
          f"gradient {row['gradient']:.2e} ({row['steps']} steps)", flush=True)


def run_fold(tx, tx_hare, y, names, names_hare, folds, fold, output_dir):
    valid = folds[fold - 1]
    train = np.concatenate([rows for k, rows in enumerate(folds, 1) if k != fold])
    x_train, x_valid = tx[train], tx[valid]
    p = tx.shape[1]
    selection_w, _ = logistic_regression(y[train], x_train, np.zeros(p), 300, 1.0)
    columns, strength = select_features(x_train, selection_w)
    cubic_columns = columns[:TOP]
    quadratic_train, quadratic_valid = quadratic_features(x_train, x_valid, columns)
    width = quadratic_train.shape[1]
    A, B, terms = append_cubic(quadratic_train, quadratic_valid,
                                x_train[:, cubic_columns], x_valid[:, cubic_columns])
    del quadratic_train, quadratic_valid, x_train, x_valid
    result = dict(fold=fold, weight_fraction=WEIGHT_FRACTION, fits=[])
    path = output_dir / f"fold_{fold}.json"
    print(f"Fold {fold}: {len(columns)} quadratic inputs, {len(cubic_columns)} cubic inputs; "
          f"{width} quadratic columns + {len(terms)} cubic terms", flush=True)
    for lambda_ in LAMBDAS:
        print(f"  Fitting quadratic, lambda={lambda_:g}", flush=True)
        w, quadratic_info = fit_quadratic(y[train], A[:, :width], p, lambda_)
        baseline_objective = None
        for degree, size in ((2, width), (3, A.shape[1])):
            info = dict(quadratic_info)
            if degree == 3:
                print(f"  Fitting cubic, lambda={lambda_:g}", flush=True)
                w, steps, gradient = fit_polynomial(
                    y[train], A, np.pad(w, (0, A.shape[1] - width)), lambda_)
                info.update(steps=int(steps), gradient=float(gradient),
                            gamma=GAMMA, max_iters=MAX_ITERS,
                            quadratic_steps=quadratic_info["steps"],
                            quadratic_gradient=quadratic_info["gradient"],
                            warm_start="same-lambda quadratic fit")
            train_scores, valid_scores = A[:, :size] @ w, B[:, :size] @ w
            info.update(
                fold=fold, degree=degree, model="quadratic" if degree == 2 else "cubic",
                lambda_=lambda_, original_columns=p, columns=size,
                weight_fraction=WEIGHT_FRACTION, selected_columns=columns.tolist(),
                selected_features=names[columns].tolist(),
                selected_strengths=strength[columns].tolist(),
                selected_strength_fraction=float(strength[columns].sum() / strength[1:].sum()),
                selection_steps=300, selection_gamma=1.0,
                cubic_columns=cubic_columns.tolist() if degree == 3 else [],
                cubic_inputs=names[cubic_columns].tolist() if degree == 3 else [],
                cubic_terms=terms if degree == 3 else [],
                threshold=float(find_best_threshold(train_scores, y[train])),
            )
            row = scored_record(y[train], y[valid], train_scores, valid_scores, info)
            row["train_objective"] = row["train"]["log_loss"] + lambda_ * float(w @ w)
            if degree == 2:
                baseline_objective = row["train_objective"]
            elif row["train_objective"] > baseline_objective + 1e-8:
                raise RuntimeError("Cubic fitting increased the training objective.")
            result["fits"].append(row)
            save(path, result)
            progress(row, f"degree {degree}, lambda={lambda_:g}")
    del A, B
    print(f"Fold {fold}: fitting quadratic + HAREHAB1, lambda=0.001", flush=True)
    result["harehab1"] = fit_evaluate(tx_hare, y, names_hare, train, valid)
    result["harehab1"]["fold"] = fold
    save(path, result)
    progress(result["harehab1"], "+ HAREHAB1")


def run_holdout(tx, tx_hare, y, names, names_hare, train, holdout, output_dir):
    result = {}
    for name, matrix, feature_names in (("without_harehab1", tx, names),
                                         ("with_harehab1", tx_hare, names_hare)):
        print(f"Final 80% fit: {name}", flush=True)
        result[name] = fit_evaluate(matrix, y, feature_names, train, holdout, "holdout")
        save(output_dir / "holdout.json", result)
        progress(result[name], name)


def aggregate(rows):
    result = {}
    for split in ("train", "validation"):
        for metric in ("f1", "precision", "recall", "accuracy", "log_loss"):
            values = [row[split][metric] for row in rows]
            result[f"{split}_{metric}_mean"] = float(np.mean(values))
            result[f"{split}_{metric}_std"] = float(np.std(values))
    result["validation_f1_t0_mean"] = float(np.mean([row["evaluation_f1_t0"] for row in rows]))
    return result


def print_comparison(rows):
    print("Degree  Lambda   CV F1 +/- std      Mean train F1  Mean CV F1(t=0)")
    for row in rows:
        cv = row["cv"]
        print(f"{row['degree']:<8}{row['lambda_']:<9g}"
              f"{cv['validation_f1_mean']:.4f} +/- {cv['validation_f1_std']:.4f}    "
              f"{cv['train_f1_mean']:.4f}         {cv['validation_f1_t0_mean']:.4f}")


def print_final_results(models):
    print("Model              CV F1 +/- std     Train F1  Holdout F1  Precision  Recall  Accuracy  Log loss")
    for row in models:
        cv, final = row["cv"], row["final"]
        h = final["holdout"]
        print(f"{row['label']:<19}{cv['validation_f1_mean']:.4f} +/- "
              f"{cv['validation_f1_std']:.4f}   {final['train']['f1']:.4f}    "
              f"{h['f1']:.4f}      {h['precision']:.4f}     {h['recall']:.4f}  "
              f"{h['accuracy']:.4f}    {h['log_loss']:.4f}")


def print_holdout_details(models):
    for row in models:
        final, h = row["final"], row["final"]["holdout"]
        print(f"{row['label']}: threshold {final['threshold']:.6f}; TP {h['tp']}, "
              f"FP {h['fp']}, FN {h['fn']}, TN {h['tn']}; "
              f"holdout F1(t=0) {final['evaluation_f1_t0']:.4f}; "
              f"{final['columns']} columns; gradient {final['gradient']:.2e} "
              f"({final['steps']} steps)")


def write_tables(result, path):
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Polynomial logistic regression.}", r"\label{tab:polynomial}",
             r"\begin{tabular}{ccccc}", r"\hline",
             r"Degree & $\lambda$ & F1 (CV) & F1 train & F1 ($t=0$) \\",
             r"\hline"]
    for row in result["comparison"]:
        cv = row["cv"]
        penalty = "0" if row["lambda_"] == 0 else "10^{-3}"
        score = f"{cv['validation_f1_mean']:.3f} \\pm {cv['validation_f1_std']:.3f}"
        if row["degree"] == 2 and row["lambda_"] == 1e-3:
            cells = [r"\textbf{2}", rf"$\mathbf{{{penalty}}}$", rf"$\mathbf{{{score}}}$",
                     rf"\textbf{{{cv['train_f1_mean']:.3f}}}",
                     rf"\textbf{{{cv['validation_f1_t0_mean']:.3f}}}"]
        else:
            cells = [str(row["degree"]), f"${penalty}$", f"${score}$",
                     f"{cv['train_f1_mean']:.3f}", f"{cv['validation_f1_t0_mean']:.3f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", "",
                  r"\begin{table}[htbp]", r"\centering",
                  r"\caption{Selected model: 3-fold CV F1 on the 80\% split and metrics on the 20\% hold-out split.}",
                  r"\label{tab:eval}", r"\begin{tabular}{lccccc}", r"\hline",
                  r" & F1 (CV) & F1 & Prec. & Rec. & Acc. \\",
                  r"\hline"])
    for row in result["models"]:
        cv, final = row["cv"], row["final"]
        h = final["holdout"]
        label = "Final model" if row["name"] == "without_harehab1" else r"+ \texttt{HAREHAB1}"
        lines.append(f"{label} & ${cv['validation_f1_mean']:.3f} \\pm "
                     f"{cv['validation_f1_std']:.3f}$ & {h['f1']:.3f} & "
                     f"{h['precision']:.3f} & {h['recall']:.3f} & {h['accuracy']:.3f} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines))


def collect(output_dir):
    folds = [json.loads((output_dir / f"fold_{k}.json").read_text()) for k in range(1, K_FOLD + 1)]
    holdout = json.loads((output_dir / "holdout.json").read_text())
    expected = {(degree, lambda_) for degree in (2, 3) for lambda_ in LAMBDAS}
    for k, fold in enumerate(folds, 1):
        if (fold["fold"] != k or fold["weight_fraction"] != WEIGHT_FRACTION
                or len(fold["fits"]) != 4
                or {(r["degree"], r["lambda_"]) for r in fold["fits"]} != expected
                or "harehab1" not in fold):
            raise ValueError(f"Fold {k} is incomplete or uses a different selection fraction.")
        if any(row.get("fold") != k for row in fold["fits"] + [fold["harehab1"]]):
            raise ValueError(f"Fold {k} contains records from a different fold.")
    records = [row for fold in folds for row in fold["fits"]]
    if set(holdout) != {"without_harehab1", "with_harehab1"}:
        raise ValueError("Both holdout model fits must be complete before collection.")
    harehab_rows = [fold["harehab1"] for fold in folds]
    settings = dict(optimizer="homework_fixed_step_gd", gamma=GAMMA,
                    max_iters=MAX_ITERS, steps=MAX_ITERS,
                    weight_fraction=WEIGHT_FRACTION, linear_steps=1000, linear_gamma=1.0)
    for row in records + harehab_rows + list(holdout.values()):
        for name, value in settings.items():
            if row.get(name) != value:
                raise ValueError(f"Mixed or stale model settings: expected {name}={value}, got {row.get(name)}.")
    for row in harehab_rows + list(holdout.values()):
        if row.get("degree") != 2 or row.get("lambda_") != 1e-3:
            raise ValueError("HAREHAB1 and holdout comparisons must use degree 2 and lambda=0.001.")
    for row in holdout.values():
        if "train" not in row or "holdout" not in row:
            raise ValueError("Incomplete holdout metrics.")
    result = dict(
        seed=SEED, k_fold=K_FOLD, holdout_ratio=HOLDOUT_RATIO, lambdas=LAMBDAS,
        weight_fraction=WEIGHT_FRACTION, top_cubic_inputs=TOP,
        optimizer="homework_fixed_step_gd", gamma=GAMMA, max_iters=MAX_ITERS,
        penalty="lambda * sum(w**2), including bias",
        selection="Preliminary linear logistic GD: 300 steps, gamma=1; smallest descending-strength prefix covering 50% of total non-bias abs(weight) * training std.",
        protocol="Outer seed-1 permutation: first 20% existing holdout; inner seed-1 permutation of the remaining 80%, split into three folds. This holdout has been evaluated previously.",
        preprocessing="Baseline preprocessing is fitted before the outer split, matching prior experiments. Selection and polynomial scaling are fitted within each training fold.",
        threshold_selection="Maximize training F1 over 100 score quantiles; apply unchanged to validation and holdout. Standard deviations use ddof=0.",
        harehab1="Raw answered indicator (~isnan), including codes 7 and 9. Target leakage comparison only; excluded from the default submission.",
        fits=records, comparison=[], models=[],
    )
    for degree in (2, 3):
        for lambda_ in LAMBDAS:
            rows = [r for r in records if r["degree"] == degree and r["lambda_"] == lambda_]
            result["comparison"].append(dict(degree=degree, lambda_=lambda_, cv=aggregate(rows)))
    for name, label in (("without_harehab1", "Quadratic"), ("with_harehab1", "+ HAREHAB1")):
        rows = ([r for r in records if r["degree"] == 2 and r["lambda_"] == 1e-3]
                if name == "without_harehab1" else [fold["harehab1"] for fold in folds])
        result["models"].append(dict(name=name, label=label, folds=rows,
                                      cv=aggregate(rows), final=holdout[name]))
    save(output_dir / "results.json", result)
    write_tables(result, output_dir / "tables.tex")
    print_comparison(result["comparison"])
    print()
    print_final_results(result["models"])
    print_holdout_details(result["models"])
    print(f"\nSaved {output_dir / 'results.json'} and {output_dir / 'tables.tex'}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", type=Path, default=ROOT / "data")
    parser.add_argument("--prepared-dir", type=Path, help="Reuse matching prepared matrices")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), help="Run one fold, including HAREHAB1")
    parser.add_argument("--holdout", action="store_true", help="Run both final 80%% fits and holdout evaluations")
    parser.add_argument("--collect", action="store_true", help="Collect saved folds and holdout; no training or data loading")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.collect:
        if args.fold is not None or args.holdout:
            parser.error("--collect cannot be combined with --fold or --holdout")
        collect(args.output_dir)
        return
    run_all = args.fold is None and not args.holdout
    tx, tx_test, y, _, names = load_features(args.data_dir, args.prepared_dir)
    raw_train, raw_test = load_harehab1(args.data_dir)
    tx_hare, tx_test_hare = add_harehab1(tx, tx_test, raw_train, raw_test)
    del tx_test, tx_test_hare, raw_train, raw_test
    names_hare = np.append(names, "HAREHAB1_answered")
    train, holdout, folds = splits(y)
    for fold in range(1, K_FOLD + 1) if run_all else ([args.fold] if args.fold is not None else []):
        run_fold(tx, tx_hare, y, names, names_hare, folds, fold, args.output_dir)
    if run_all or args.holdout:
        run_holdout(tx, tx_hare, y, names, names_hare, train, holdout, args.output_dir)
    if run_all:
        collect(args.output_dir)


if __name__ == "__main__":
    main()
