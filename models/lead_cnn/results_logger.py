"""
results_logger.py — Printing and saving utilities for LEAD-CNN runs.

Functions:
  print_model_summary(model)          — formatted parameter table
  print_training_history(history)     — epoch-by-epoch table
  print_final_scores(y_true, y_pred)  — accuracy, precision, recall, F1, Kappa
  print_confusion_matrix(...)         — formatted confusion matrix
  save_run_results(...)               — save metrics as JSON + CSV
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


# ── Model summary ────────────

def print_model_summary(model):
    """
    Prints a clean layer-by-layer parameter table matching the paper's Table 2 style.
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  LEAD-CNN — Layer Parameter Table")
    print(sep)
    print(f"  {'Layer Name':<30} {'Layer Type':<22} {'Output Shape':<18} {'Params':>8}")
    print(f"  {'-'*30} {'-'*22} {'-'*18} {'-'*8}")

    total_params = 0
    for layer in model.layers:
        name = layer.name
        ltype = type(layer).__name__
        try:
            shape = str(layer.output.shape[1:])
        except Exception:
            shape = "N/A"
        params = layer.count_params()
        total_params += params
        print(f"  {name:<30} {ltype:<22} {shape:<18} {params:>8,}")

    print(f"  {'-'*30} {'-'*22} {'-'*18} {'-'*8}")
    print(f"  {'TOTAL PARAMETERS':<52} {total_params:>8,}")
    print(f"  {'Paper target: ~1,132,612':<52}")
    print(sep + "\n")


# ── Training history ────────────

def print_training_history(history):
    """
    Prints epoch-by-epoch train/val accuracy and loss in a table.
    Useful for comparing multiple training runs at a glance.
    """
    h = history.history
    epochs = range(1, len(h['accuracy']) + 1)

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Training History")
    print(sep)
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

    best_val_acc = max(h['val_accuracy'])
    for ep in epochs:
        i = ep - 1
        marker = " <-- best" if h['val_accuracy'][i] == best_val_acc else ""
        print(f"  {ep:>6}  {h['loss'][i]:>12.4f}  {h['accuracy'][i]:>10.4f}  "
              f"{h['val_loss'][i]:>10.4f}  {h['val_accuracy'][i]:>10.4f}{marker}")

    print(sep)
    print(f"  Best val accuracy: {best_val_acc:.4f} "
          f"(epoch {h['val_accuracy'].index(best_val_acc) + 1})")
    print(sep + "\n")


# ── Final evaluation scores ─────────────

def print_final_scores(y_true, y_pred, class_labels):
    """
    Prints overall accuracy, weighted precision/recall/F1, Cohen Kappa,
    and per-class metrics — matching the paper's Table 4/5 format.
    """
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='weighted') * 100
    rec = recall_score(y_true, y_pred, average='weighted') * 100
    f1 = f1_score(y_true, y_pred, average='weighted') * 100
    kappa = cohen_kappa_score(y_true, y_pred)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  LEAD-CNN — Final Evaluation Results")
    print(sep)
    print(f"  {'Metric':<25} {'Score':>10}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'Accuracy':<25} {acc:>9.2f}%")
    print(f"  {'Precision (weighted)':<25} {prec:>9.2f}%")
    print(f"  {'Recall (weighted)':<25} {rec:>9.2f}%")
    print(f"  {'F1-Score (weighted)':<25} {f1:>9.2f}%")
    print(f"  {'Cohen Kappa':<25} {kappa:>10.4f}")
    print(sep)

    # Paper target comparison
    print(f"\n  Paper targets:  Acc=98.70  Prec=98.65  Rec=98.60  F1=98.62  Kappa=0.9825")
    print(f"  Your results:   Acc={acc:.2f}  Prec={prec:.2f}  Rec={rec:.2f}  "
          f"F1={f1:.2f}  Kappa={kappa:.4f}")

    # Per-class breakdown
    print(f"\n{sep}")
    print(f"  Per-Class Breakdown")
    print(sep)
    report = classification_report(
        y_true, y_pred, target_names=class_labels, digits=4
    )
    print(report)
    print(sep + "\n")


# ── Confusion matrix ──────────

def print_confusion_matrix(y_true, y_pred, class_labels):
    """
    Prints a readable confusion matrix with class labels on both axes.
    """
    cm = confusion_matrix(y_true, y_pred)
    col_width = max(len(c) for c in class_labels) + 2

    sep = "=" * (col_width * (len(class_labels) + 1) + 4)
    print(f"\n{sep}")
    print(f"  Confusion Matrix  (rows=Actual, cols=Predicted)")
    print(sep)

    # Header row
    header = f"  {'':>{col_width}}"
    for label in class_labels:
        header += f"  {label:>{col_width}}"
    print(header)
    print(f"  {'-'*(col_width*(len(class_labels)+1)+2)}")

    for i, label in enumerate(class_labels):
        row = f"  {label:>{col_width}}"
        for j in range(len(class_labels)):
            val = cm[i][j]
            marker = f"[{val}]" if i == j else f" {val} "
            row += f"  {marker:>{col_width}}"
        print(row)

    print(sep + "\n")
    return cm


# ── Save results ──────────────

def save_run_results(y_true, y_pred, class_labels, results_dir):
    """
    Saves a JSON and CSV file for this evaluation run.
    Files are timestamped so multiple runs don't overwrite each other.

    Saved to:
      results/run_YYYYMMDD_HHMMSS_metrics.json
      results/run_YYYYMMDD_HHMMSS_confusion.csv
      results/all_runs_summary.csv  (appended each run)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='weighted') * 100
    rec = recall_score(y_true, y_pred, average='weighted') * 100
    f1 = f1_score(y_true, y_pred, average='weighted') * 100
    kappa = cohen_kappa_score(y_true, y_pred)

    # ── Per-class metrics ──
    per_class = {}
    for i, label in enumerate(class_labels):
        mask = (y_true == i)
        tp = int(np.sum((y_pred == i) & mask))
        fp = int(np.sum((y_pred == i) & ~mask))
        fn = int(np.sum((y_pred != i) & mask))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[label] = {
            "precision": round(p * 100, 4),
            "recall": round(r * 100, 4),
            "f1": round(f * 100, 4),
        }

    metrics = {
        "timestamp": timestamp,
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "f1_weighted": round(f1, 4),
        "cohen_kappa": round(kappa, 6),
        "per_class": per_class,
        "paper_targets": {
            "accuracy": 98.70,
            "precision": 98.65,
            "recall": 98.60,
            "f1": 98.62,
            "cohen_kappa": 0.9825,
        },
    }

    # Save JSON
    json_path = os.path.join(results_dir, f"run_{timestamp}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix CSV
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(results_dir, f"run_{timestamp}_confusion.csv")
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_labels)
        for i, label in enumerate(class_labels):
            writer.writerow([label] + cm[i].tolist())

    # Append to rolling summary CSV
    summary_path = os.path.join(results_dir, "all_runs_summary.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "accuracy", "precision_weighted",
                "recall_weighted", "f1_weighted", "cohen_kappa"
            ])
        writer.writerow([
            timestamp,
            round(acc, 4), round(prec, 4),
            round(rec, 4), round(f1, 4),
            round(kappa, 6),
        ])

    print(f"  Results saved:")
    print(f"    Metrics JSON:       {json_path}")
    print(f"    Confusion CSV:      {cm_path}")
    print(f"    All runs summary:   {summary_path}\n")
