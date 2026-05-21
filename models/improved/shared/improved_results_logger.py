"""
Shared results logger for all improved LEAD-CNN variants.
Same interface as lead_cnn/results_logger.py and baseline_results_logger.py.
Also writes to results/improved/all_improved_summary.csv for cross-variant comparison.
"""

import os
import json
import csv
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Paper target for comparison
PAPER_TARGET = {
    "accuracy": 98.70,
    "precision": 98.65,
    "recall": 98.60,
    "f1": 98.62,
    "cohen_kappa": 0.9825,
}


# ── Model summary ─────────────────────────────────────────────────────────────

def print_model_summary(model, variant_name):
    total     = model.count_params()
    trainable = sum(l.count_params() for l in model.layers if l.trainable)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {variant_name.upper()} — Parameter Summary")
    print(sep)
    print(f"  {'Total parameters':<30} {total:>12,}")
    print(f"  {'Trainable':<30} {trainable:>12,}")
    print(f"  Paper target: ~1,132,612")
    print(sep + "\n")


# ── Training history ──────────────────────────────────────────────────────────

def print_training_history(history, variant_name):
    h      = history.history
    epochs = range(1, len(h['accuracy']) + 1)
    best   = max(h['val_accuracy'])

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  {variant_name.upper()} — Training History")
    print(sep)
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

    for ep in epochs:
        i = ep - 1
        marker = " <-- best" if h['val_accuracy'][i] == best else ""
        print(f"  {ep:>6}  {h['loss'][i]:>12.4f}  {h['accuracy'][i]:>10.4f}  "
              f"{h['val_loss'][i]:>10.4f}  {h['val_accuracy'][i]:>10.4f}{marker}")

    print(sep)
    print(f"  Best val accuracy: {best:.4f} "
          f"(epoch {h['val_accuracy'].index(best) + 1})")
    print(sep + "\n")


# ── K-Fold history (fold-by-fold summary) ────────────────────────────────────

def print_kfold_summary(fold_results, variant_name):
    """
    fold_results: list of dicts with keys:
        fold, accuracy, precision, recall, f1, cohen_kappa
    """
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  {variant_name.upper()} — K-Fold Cross Validation Summary")
    print(sep)
    print(f"  {'Fold':>6}  {'Accuracy':>10}  {'Precision':>10}  "
          f"{'Recall':>10}  {'F1':>10}  {'Kappa':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    accs   = [r['accuracy']    for r in fold_results]
    precs  = [r['precision']   for r in fold_results]
    recs   = [r['recall']      for r in fold_results]
    f1s    = [r['f1']          for r in fold_results]
    kappas = [r['cohen_kappa'] for r in fold_results]

    for r in fold_results:
        print(f"  {r['fold']:>6}  {r['accuracy']:>9.2f}%  {r['precision']:>9.2f}%  "
              f"{r['recall']:>9.2f}%  {r['f1']:>9.2f}%  {r['cohen_kappa']:>10.4f}")

    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'Mean':>6}  {np.mean(accs):>9.2f}%  {np.mean(precs):>9.2f}%  "
          f"{np.mean(recs):>9.2f}%  {np.mean(f1s):>9.2f}%  {np.mean(kappas):>10.4f}")
    print(f"  {'Std':>6}  {np.std(accs):>9.2f}%  {np.std(precs):>9.2f}%  "
          f"{np.std(recs):>9.2f}%  {np.std(f1s):>9.2f}%  {np.std(kappas):>10.4f}")
    print(sep + "\n")


# ── Final scores ──────────────────────────────────────────────────────────────

def print_final_scores(y_true, y_pred, class_labels, variant_name):
    acc   = accuracy_score(y_true, y_pred) * 100
    prec  = precision_score(y_true, y_pred, average='weighted') * 100
    rec   = recall_score(y_true, y_pred, average='weighted') * 100
    f1    = f1_score(y_true, y_pred, average='weighted') * 100
    kappa = cohen_kappa_score(y_true, y_pred)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {variant_name.upper()} — Final Evaluation Results")
    print(sep)
    print(f"  {'Metric':<25} {'Score':>10}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'Accuracy':<25} {acc:>9.2f}%")
    print(f"  {'Precision (weighted)':<25} {prec:>9.2f}%")
    print(f"  {'Recall (weighted)':<25} {rec:>9.2f}%")
    print(f"  {'F1-Score (weighted)':<25} {f1:>9.2f}%")
    print(f"  {'Cohen Kappa':<25} {kappa:>10.4f}")
    print(sep)

    t = PAPER_TARGET
    print(f"\n  Paper (LEAD-CNN): Acc={t['accuracy']}  Prec={t['precision']}  "
          f"Rec={t['recall']}  F1={t['f1']}  Kappa={t['cohen_kappa']}")
    print(f"  {variant_name:<17}: Acc={acc:.2f}  Prec={prec:.2f}  "
          f"Rec={rec:.2f}  F1={f1:.2f}  Kappa={kappa:.4f}")

    print(f"\n{sep}")
    print(f"  Per-Class Breakdown")
    print(sep)
    print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))
    print(sep + "\n")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def print_confusion_matrix(y_true, y_pred, class_labels, variant_name):
    cm        = confusion_matrix(y_true, y_pred)
    col_width = max(len(c) for c in class_labels) + 2

    sep = "=" * (col_width * (len(class_labels) + 1) + 4)
    print(f"\n{sep}")
    print(f"  {variant_name.upper()} — Confusion Matrix  (rows=Actual, cols=Predicted)")
    print(sep)

    header = f"  {'':>{col_width}}"
    for label in class_labels:
        header += f"  {label:>{col_width}}"
    print(header)
    print(f"  {'-'*(col_width*(len(class_labels)+1)+2)}")

    for i, label in enumerate(class_labels):
        row = f"  {label:>{col_width}}"
        for j in range(len(class_labels)):
            val    = cm[i][j]
            marker = f"[{val}]" if i == j else f" {val} "
            row   += f"  {marker:>{col_width}}"
        print(row)

    print(sep + "\n")
    return cm


# ── Save results ──────────────────────────────────────────────────────────────

def save_run_results(y_true, y_pred, class_labels, variant_name,
                     results_dir, extra_metrics=None):
    """
    Saves timestamped JSON + confusion CSV for this run.
    Appends a row to results/improved/all_improved_summary.csv.

    extra_metrics: optional dict of additional values to include in JSON
                   (e.g. kfold mean/std, class_weights used)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(results_dir, exist_ok=True)

    acc   = accuracy_score(y_true, y_pred) * 100
    prec  = precision_score(y_true, y_pred, average='weighted') * 100
    rec   = recall_score(y_true, y_pred, average='weighted') * 100
    f1    = f1_score(y_true, y_pred, average='weighted') * 100
    kappa = cohen_kappa_score(y_true, y_pred)

    per_class = {}
    for i, label in enumerate(class_labels):
        mask = (y_true == i)
        tp   = int(np.sum((y_pred == i) & mask))
        fp   = int(np.sum((y_pred == i) & ~mask))
        fn   = int(np.sum((y_pred != i) & mask))
        p    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f    = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[label] = {
            "precision": round(p * 100, 4),
            "recall":    round(r * 100, 4),
            "f1":        round(f * 100, 4),
        }

    metrics = {
        "timestamp":          timestamp,
        "variant":            variant_name,
        "accuracy":           round(acc,   4),
        "precision_weighted": round(prec,  4),
        "recall_weighted":    round(rec,   4),
        "f1_weighted":        round(f1,    4),
        "cohen_kappa":        round(kappa, 6),
        "per_class":          per_class,
        "paper_target":       PAPER_TARGET,
    }

    if extra_metrics:
        metrics.update(extra_metrics)

    # JSON
    json_path = os.path.join(results_dir, f"run_{timestamp}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion CSV
    cm     = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(results_dir, f"run_{timestamp}_confusion.csv")
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_labels)
        for i, label in enumerate(class_labels):
            writer.writerow([label] + cm[i].tolist())

    # Rolling improved summary
    import sys, os as _os
    _here    = _os.path.dirname(_os.path.abspath(__file__))
    _root    = _os.path.abspath(_os.path.join(_here, "..", "..", ".."))
    summary_path  = _os.path.join(_root, "results", "improved", "all_improved_summary.csv")
    _os.makedirs(_os.path.dirname(summary_path), exist_ok=True)
    write_header  = not _os.path.exists(summary_path)

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "variant", "accuracy", "precision_weighted",
                "recall_weighted", "f1_weighted", "cohen_kappa"
            ])
        writer.writerow([
            timestamp, variant_name,
            round(acc, 4), round(prec, 4),
            round(rec,  4), round(f1,   4),
            round(kappa, 6),
        ])

    print(f"  Results saved:")
    print(f"    Metrics JSON:       {json_path}")
    print(f"    Confusion CSV:      {cm_path}")
    print(f"    Improved summary:   {summary_path}\n")
