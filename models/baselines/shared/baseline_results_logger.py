"""
Shared results logger for all baseline models.
Mirrors results_logger.py in lead_cnn but also writes to a
combined baselines summary CSV for easy cross-model comparison.
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

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_config import RESULTS_DIR


# Paper Table 5 targets for quick reference during evaluation
PAPER_TARGETS = {
    "densenet201":    {"accuracy": 96.10, "precision": 95.84, "recall": 95.79, "f1": 95.81},
    "resnet101":      {"accuracy": 97.17, "precision": 97.11, "recall": 96.93, "f1": 96.98},
    "mobilenetv1":    {"accuracy": 94.73, "precision": 94.47, "recall": 94.30, "f1": 94.37},
    "xception":       {"accuracy": 92.75, "precision": 92.29, "recall": 92.20, "f1": 92.20},
    "efficientnetb4": {"accuracy": 94.50, "precision": 94.10, "recall": 94.10, "f1": 94.08},
    "vgg19":          {"accuracy": 96.94, "precision": 96.81, "recall": 96.68, "f1": 96.72},
}


# ── Model summary ─────────────────────────────────────────────────────────────

def print_model_summary(model, model_name):
    trainable     = sum(layer.count_params() for layer in model.layers if layer.trainable)
    non_trainable = sum(layer.count_params() for layer in model.layers if not layer.trainable)
    total         = model.count_params()

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {model_name.upper()} — Parameter Summary")
    print(sep)
    print(f"  {'Total parameters':<30} {total:>12,}")
    print(f"  {'Trainable (head only)':<30} {trainable:>12,}")
    print(f"  {'Non-trainable (backbone)':<30} {non_trainable:>12,}")
    print(sep + "\n")


# ── Training history ──────────────────────────────────────────────────────────

def print_training_history(history, model_name):
    h = history.history
    epochs = range(1, len(h['accuracy']) + 1)
    best_val_acc = max(h['val_accuracy'])

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  {model_name.upper()} — Training History")
    print(sep)
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

    for ep in epochs:
        i = ep - 1
        marker = " <-- best" if h['val_accuracy'][i] == best_val_acc else ""
        print(f"  {ep:>6}  {h['loss'][i]:>12.4f}  {h['accuracy'][i]:>10.4f}  "
              f"{h['val_loss'][i]:>10.4f}  {h['val_accuracy'][i]:>10.4f}{marker}")

    print(sep)
    print(f"  Best val accuracy: {best_val_acc:.4f} "
          f"(epoch {h['val_accuracy'].index(best_val_acc) + 1})")
    print(sep + "\n")


# ── Final scores ──────────────────────────────────────────────────────────────

def print_final_scores(y_true, y_pred, class_labels, model_name):
    acc  = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='weighted') * 100
    rec  = recall_score(y_true, y_pred, average='weighted') * 100
    f1   = f1_score(y_true, y_pred, average='weighted') * 100
    kappa = cohen_kappa_score(y_true, y_pred)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {model_name.upper()} — Final Evaluation Results")
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
    key = model_name.lower().replace("-", "").replace("_", "")
    if key in PAPER_TARGETS:
        t = PAPER_TARGETS[key]
        print(f"\n  Paper targets:  Acc={t['accuracy']}  Prec={t['precision']}  "
              f"Rec={t['recall']}  F1={t['f1']}")
        print(f"  Your results:   Acc={acc:.2f}  Prec={prec:.2f}  "
              f"Rec={rec:.2f}  F1={f1:.2f}  Kappa={kappa:.4f}")

    # Per-class breakdown
    print(f"\n{sep}")
    print(f"  Per-Class Breakdown")
    print(sep)
    print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))
    print(sep + "\n")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def print_confusion_matrix(y_true, y_pred, class_labels, model_name):
    cm = confusion_matrix(y_true, y_pred)
    col_width = max(len(c) for c in class_labels) + 2

    sep = "=" * (col_width * (len(class_labels) + 1) + 4)
    print(f"\n{sep}")
    print(f"  {model_name.upper()} — Confusion Matrix  (rows=Actual, cols=Predicted)")
    print(sep)

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


# ── Save results ──────────────────────────────────────────────────────────────

def save_run_results(y_true, y_pred, class_labels, model_name):
    """
    Saves per-run JSON + confusion CSV into results/baselines/<model_name>/
    and appends a row to results/baselines/all_baselines_summary.csv.
    """
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir     = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    acc   = accuracy_score(y_true, y_pred) * 100
    prec  = precision_score(y_true, y_pred, average='weighted') * 100
    rec   = recall_score(y_true, y_pred, average='weighted') * 100
    f1    = f1_score(y_true, y_pred, average='weighted') * 100
    kappa = cohen_kappa_score(y_true, y_pred)

    per_class = {}
    for i, label in enumerate(class_labels):
        mask = (y_true == i)
        tp = int(np.sum((y_pred == i) & mask))
        fp = int(np.sum((y_pred == i) & ~mask))
        fn = int(np.sum((y_pred != i) & mask))
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[label] = {
            "precision": round(p * 100, 4),
            "recall":    round(r * 100, 4),
            "f1":        round(f * 100, 4),
        }

    key = model_name.lower().replace("-", "").replace("_", "")
    metrics = {
        "timestamp":          timestamp,
        "model":              model_name,
        "accuracy":           round(acc,   4),
        "precision_weighted": round(prec,  4),
        "recall_weighted":    round(rec,   4),
        "f1_weighted":        round(f1,    4),
        "cohen_kappa":        round(kappa, 6),
        "per_class":          per_class,
        "paper_targets":      PAPER_TARGETS.get(key, {}),
    }

    # JSON
    json_path = os.path.join(model_dir, f"run_{timestamp}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion CSV
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(model_dir, f"run_{timestamp}_confusion.csv")
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_labels)
        for i, label in enumerate(class_labels):
            writer.writerow([label] + cm[i].tolist())

    # Rolling summary for all baselines
    summary_path = os.path.join(RESULTS_DIR, "all_baselines_summary.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "model", "accuracy", "precision_weighted",
                "recall_weighted", "f1_weighted", "cohen_kappa"
            ])
        writer.writerow([
            timestamp, model_name,
            round(acc, 4), round(prec, 4),
            round(rec, 4), round(f1, 4),
            round(kappa, 6),
        ])

    print(f"  Results saved:")
    print(f"    Metrics JSON:    {json_path}")
    print(f"    Confusion CSV:   {cm_path}")
    print(f"    Baselines CSV:   {summary_path}\n")
