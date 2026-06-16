"""
Shared results logger for architecture experiments.
Same format as other loggers. Writes to results/architecture_experiments/
and appends to a shared all_arch_summary.csv for cross-experiment comparison.
"""

import os
import sys
import json
import csv
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score,
    accuracy_score, precision_score, recall_score, f1_score,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from arch_config import ARCH_RESULTS_DIR, PROJECT_ROOT

# Reference scores for comparison
PAPER_TARGET = {
    "accuracy": 98.70, "precision": 98.65,
    "recall": 98.60,   "f1": 98.62, "cohen_kappa": 0.9825,
}
BASE_LEAD_CNN = {
    "accuracy": 98.58, "precision": 98.58,
    "recall": 98.58,   "f1": 98.58, "cohen_kappa": 0.9810,
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
    print(f"  {'Base LEAD-CNN (paper)':<30} {'~1,132,612':>12}")
    diff = total - 1132612
    sign = "+" if diff >= 0 else ""
    print(f"  {'Difference':<30} {sign}{diff:>11,}")
    print(sep + "\n")


# ── Training history ──────────────────────────────────────────────────────────

def print_training_history(history, variant_name):
    h    = history.history
    eps  = range(1, len(h['accuracy']) + 1)
    best = max(h['val_accuracy'])

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  {variant_name.upper()} — Training History")
    print(sep)
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    for ep in eps:
        i = ep - 1
        marker = " <-- best" if h['val_accuracy'][i] == best else ""
        print(f"  {ep:>6}  {h['loss'][i]:>12.4f}  {h['accuracy'][i]:>10.4f}  "
              f"{h['val_loss'][i]:>10.4f}  {h['val_accuracy'][i]:>10.4f}{marker}")
    print(sep)
    print(f"  Best val accuracy : {best:.4f} "
          f"(epoch {h['val_accuracy'].index(best)+1})")
    print(sep + "\n")


# ── Final scores ──────────────────────────────────────────────────────────────

def print_final_scores(y_true, y_pred, class_labels, variant_name, config=None):
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

    # Comparison table
    print(f"\n  {'Model':<25} {'Acc':>8}  {'F1':>8}  {'Kappa':>8}")
    print(f"  {'-'*25} {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Paper (LEAD-CNN)':<25} {PAPER_TARGET['accuracy']:>7.2f}%  "
          f"{PAPER_TARGET['f1']:>7.2f}%  {PAPER_TARGET['cohen_kappa']:>8.4f}")
    print(f"  {'Base (replicated)':<25} {BASE_LEAD_CNN['accuracy']:>7.2f}%  "
          f"{BASE_LEAD_CNN['f1']:>7.2f}%  {BASE_LEAD_CNN['cohen_kappa']:>8.4f}")
    print(f"  {variant_name:<25} {acc:>7.2f}%  {f1:>7.2f}%  {kappa:>8.4f}")

    # Print config used if provided
    if config is not None:
        print(f"\n  Config used:")
        for attr in ['LEARNING_RATE', 'BATCH_SIZE', 'EPOCHS',
                     'WIDTH_MULTIPLIER', 'SE_REDUCTION_RATIO',
                     'DROPOUT_CONV', 'DROPOUT_FC1', 'DROPOUT_FC2']:
            if hasattr(config, attr):
                print(f"    {attr:<25} {getattr(config, attr)}")

    print(f"\n{sep}")
    print(f"  Per-Class Breakdown")
    print(sep)
    print(classification_report(
        y_true, y_pred, target_names=class_labels, digits=4
    ))
    print(sep + "\n")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def print_confusion_matrix(y_true, y_pred, class_labels, variant_name):
    cm        = confusion_matrix(y_true, y_pred)
    col_width = max(len(c) for c in class_labels) + 2

    sep = "=" * (col_width * (len(class_labels) + 1) + 4)
    print(f"\n{sep}")
    print(f"  {variant_name.upper()} — Confusion Matrix")
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
                     results_dir, config=None, extra_metrics=None):
    """
    Saves timestamped JSON + confusion CSV.
    Appends to results/architecture_experiments/all_arch_summary.csv.
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

    # Capture config snapshot so every run is fully reproducible
    config_snapshot = {}
    if config is not None:
        for attr in ['LEARNING_RATE', 'BATCH_SIZE', 'EPOCHS',
                     'WIDTH_MULTIPLIER', 'SE_REDUCTION_RATIO',
                     'DROPOUT_CONV', 'DROPOUT_FC1', 'DROPOUT_FC2',
                     'LEAKY_ALPHA']:
            if hasattr(config, attr):
                config_snapshot[attr] = getattr(config, attr)

    metrics = {
        "timestamp":          timestamp,
        "variant":            variant_name,
        "accuracy":           round(acc,   4),
        "precision_weighted": round(prec,  4),
        "recall_weighted":    round(rec,   4),
        "f1_weighted":        round(f1,    4),
        "cohen_kappa":        round(kappa, 6),
        "per_class":          per_class,
        "config_used":        config_snapshot,
        "paper_target":       PAPER_TARGET,
        "base_lead_cnn":      BASE_LEAD_CNN,
    }
    if extra_metrics:
        metrics.update(extra_metrics)

    json_path = os.path.join(results_dir, f"run_{timestamp}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    cm      = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(results_dir, f"run_{timestamp}_confusion.csv")
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_labels)
        for i, label in enumerate(class_labels):
            writer.writerow([label] + cm[i].tolist())

    summary_path = os.path.join(
        PROJECT_ROOT, "results", "architecture_experiments", "all_arch_summary.csv"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "variant", "accuracy", "precision_weighted",
                "recall_weighted", "f1_weighted", "cohen_kappa",
                "learning_rate", "batch_size", "epochs",
                "width_multiplier", "se_reduction_ratio",
            ])
        writer.writerow([
            timestamp, variant_name,
            round(acc, 4), round(prec, 4), round(rec, 4),
            round(f1, 4),  round(kappa, 6),
            config_snapshot.get('LEARNING_RATE', ''),
            config_snapshot.get('BATCH_SIZE', ''),
            config_snapshot.get('EPOCHS', ''),
            config_snapshot.get('WIDTH_MULTIPLIER', ''),
            config_snapshot.get('SE_REDUCTION_RATIO', ''),
        ])

    print(f"  Results saved:")
    print(f"    Metrics  : {json_path}")
    print(f"    Confusion: {cm_path}")
    print(f"    Summary  : {summary_path}\n")


# ── ROC / AUC ─────────────────────────────────────────────────────────────────

def plot_roc_curves(y_true, y_prob, class_labels, variant_name, results_dir):
    """
    Plots one ROC curve per class (one-vs-rest) plus the macro-average.
    Saves the figure and returns a dict of per-class AUC values.

    Args:
        y_true:       (N,) int array of true class indices
        y_prob:       (N, C) float array of softmax probabilities
        class_labels: list of class name strings
        variant_name: string identifier for titles and filenames
        results_dir:  directory to save the figure and AUC JSON
    """
    try:
        import matplotlib
        matplotlib.use('Agg')   # non-interactive backend — works on remote servers
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
    except ImportError as e:
        print(f"  Warning: ROC plot skipped — missing dependency: {e}")
        return {}

    n_classes = len(class_labels)
    # Binarize true labels for one-vs-rest
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    auc_scores = {}

    # Per-class curves
    for i, (label, color) in enumerate(zip(class_labels, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores[label] = round(float(roc_auc), 6)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label} (AUC = {roc_auc:.4f})")

    # Macro-average curve
    all_fpr = np.unique(np.concatenate(
        [roc_curve(y_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)]
    ))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    auc_scores['macro_average'] = round(float(macro_auc), 6)

    ax.plot(all_fpr, mean_tpr, color='black', lw=2.5, linestyle='--',
            label=f"Macro Average (AUC = {macro_auc:.4f})")

    # Reference line
    ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves — {variant_name}', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(results_dir, exist_ok=True)
    fig_path = os.path.join(results_dir, f"roc_curves_{variant_name}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # Save AUC scores to JSON
    auc_path = os.path.join(results_dir, f"auc_scores_{variant_name}.json")
    with open(auc_path, 'w') as f:
        json.dump({"variant": variant_name, "auc_scores": auc_scores}, f, indent=2)

    # Print summary
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {variant_name.upper()} — AUC Scores")
    print(sep)
    for label, score in auc_scores.items():
        print(f"  {label:<20} {score:.4f}")
    print(sep)
    print(f"  ROC figure : {fig_path}")
    print(f"  AUC JSON   : {auc_path}\n")

    return auc_scores


# ── Save predictions for McNemar's test ───────────────────────────────────────

def save_predictions(y_true, y_pred, y_prob, variant_name, results_dir):
    """
    Saves raw per-sample predictions to a .npz file so McNemar's test
    can load and compare them against another model's predictions later.

    Saved to: results_dir/predictions_<variant_name>.npz
    """
    os.makedirs(results_dir, exist_ok=True)
    pred_path = os.path.join(results_dir, f"predictions_{variant_name}.npz")
    np.savez(pred_path,
             y_true=y_true,
             y_pred=y_pred,
             y_prob=y_prob,
             variant=np.array([variant_name]))
    print(f"  Predictions saved: {pred_path}")
    return pred_path
