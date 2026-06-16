"""
roc_curves.py — Standalone ROC/AUC generator.

Loads any saved .keras model, runs it on the test set,
and produces ROC curves + AUC scores.
Axes are zoomed to FPR 0-0.05 / TPR 0.95-1.0 by default
so differences between near-perfect models are visible.

Usage:
    # Single model
    python models/architecture_experiments/shared/roc_curves.py \
        --model results/architecture_experiments/dsc_dimred/dsc_dimred_best.keras \
        --variant dsc_dimred

    # With comparison overlay
    python models/architecture_experiments/shared/roc_curves.py \
        --model results/architecture_experiments/dsc_dimred/dsc_dimred_best.keras \
        --variant dsc_dimred \
        --compare results/predictions_lead_cnn.npz \
        --compare-name lead_cnn
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

sys.path.insert(0, _HERE)

from arch_config   import DATASET_PATH, IMG_SIZE, RANDOM_SEED
from arch_dataset  import create_generators
from arch_results_logger import save_predictions

# ── Zoom settings ─────────────────────────────────────────────────────────────
# Both axes zoomed to where near-perfect models actually differ.
# Change these if you want the full 0-1 range.
FPR_MAX = 0.05    # x-axis upper limit
TPR_MIN = 0.95    # y-axis lower limit


def _plot_roc_zoomed(ax, fpr, tpr, roc_auc, label, color, linestyle='-', lw=2):
    """Plots a single ROC curve on ax with zoom applied."""
    ax.plot(fpr, tpr, color=color, lw=lw, linestyle=linestyle,
            label=f"{label} (AUC = {roc_auc:.4f})")


def plot_single_model_roc(y_true, y_prob, class_labels, variant_name, output_dir):
    """
    Per-class ROC curves for a single model, zoomed to top-left corner.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        import json
    except ImportError as e:
        print(f"  Warning: ROC plot skipped — {e}")
        return {}

    n_classes = len(class_labels)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))
    colors    = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'ROC Curves — {variant_name}', fontsize=13)

    auc_scores = {}

    for i, (label, color) in enumerate(zip(class_labels, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores[label] = round(float(roc_auc), 6)
        # Full range plot (left)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{label} (AUC = {roc_auc:.4f})")
        # Zoomed plot (right)
        axes[1].plot(fpr, tpr, color=color, lw=2,
                     label=f"{label} (AUC = {roc_auc:.4f})")

    # Macro average
    all_fpr  = np.unique(np.concatenate(
        [roc_curve(y_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)]
    ))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        mean_tpr   += np.interp(all_fpr, fpr, tpr)
    mean_tpr   /= n_classes
    macro_auc   = auc(all_fpr, mean_tpr)
    auc_scores['macro_average'] = round(float(macro_auc), 6)

    for ax in axes:
        ax.plot(all_fpr, mean_tpr, color='black', lw=2.5, linestyle='--',
                label=f"Macro Average (AUC = {macro_auc:.4f})")
        ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.4)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.02])
    axes[0].set_title('Full Range', fontsize=11)

    axes[1].set_xlim([0.0, FPR_MAX])
    axes[1].set_ylim([TPR_MIN, 1.002])
    axes[1].set_title(f'Zoomed (FPR ≤ {FPR_MAX}, TPR ≥ {TPR_MIN})', fontsize=11)

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"roc_curves_{variant_name}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save AUC JSON
    auc_path = os.path.join(output_dir, f"auc_scores_{variant_name}.json")
    import json
    with open(auc_path, 'w') as f:
        json.dump({"variant": variant_name, "auc_scores": auc_scores}, f, indent=2)

    # Print summary
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {variant_name.upper()} — AUC Scores")
    print(sep)
    for label, score in auc_scores.items():
        print(f"  {label:<20} {score:.6f}")
    print(sep)
    print(f"  ROC figure : {fig_path}")
    print(f"  AUC JSON   : {auc_path}\n")

    return auc_scores


def plot_comparison_roc(pred_path_a, name_a, pred_path_b, name_b, output_dir):
    """
    Macro-average ROC comparison — full range AND zoomed side by side.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
    except ImportError as e:
        print(f"  Warning: comparison plot skipped — {e}")
        return

    def load_macro(path):
        data  = np.load(path, allow_pickle=True)
        y_t   = data['y_true']
        y_p   = data['y_prob']
        n_cls = y_p.shape[1]
        y_bin = label_binarize(y_t, classes=list(range(n_cls)))
        all_fpr = np.unique(np.concatenate(
            [roc_curve(y_bin[:, i], y_p[:, i])[0] for i in range(n_cls)]
        ))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_cls):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_p[:, i])
            mean_tpr   += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= n_cls
        return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

    fpr_a, tpr_a, auc_a = load_macro(pred_path_a)
    fpr_b, tpr_b, auc_b = load_macro(pred_path_b)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Macro-Average ROC Comparison: {name_a} vs {name_b}', fontsize=13)

    for ax in axes:
        ax.plot(fpr_a, tpr_a, color='#377eb8', lw=2.5,
                label=f"{name_a}  (AUC = {auc_a:.4f})")
        ax.plot(fpr_b, tpr_b, color='#e41a1c', lw=2.5, linestyle='--',
                label=f"{name_b}  (AUC = {auc_b:.4f})")
        ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.4)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.02])
    axes[0].set_title('Full Range', fontsize=11)

    axes[1].set_xlim([0.0, FPR_MAX])
    axes[1].set_ylim([TPR_MIN, 1.002])
    axes[1].set_title(f'Zoomed (FPR ≤ {FPR_MAX}, TPR ≥ {TPR_MIN})', fontsize=11)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"roc_comparison_{name_a}_vs_{name_b}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison ROC saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ROC/AUC curves for a saved LEAD-CNN variant."
    )
    parser.add_argument("--model",        required=True,
                        help="Path to .keras model file.")
    parser.add_argument("--variant",      required=True,
                        help="Name for this model.")
    parser.add_argument("--output",       default=None,
                        help="Output directory. Defaults to results/roc/")
    parser.add_argument("--compare",      default=None,
                        help="Path to a .npz predictions file for comparison.")
    parser.add_argument("--compare-name", default="comparison",
                        help="Name for the comparison model.")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(_ROOT, "results", "roc")
    os.makedirs(output_dir, exist_ok=True)

    _, _, test_gen = create_generators()
    model = tf.keras.models.load_model(args.model, compile=False)

    print(f"\nRunning predictions for {args.variant}...")
    y_prob       = model.predict(test_gen, verbose=1)
    y_pred       = np.argmax(y_prob, axis=1)
    y_true       = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    plot_single_model_roc(y_true, y_prob, class_labels, args.variant, output_dir)
    pred_path = save_predictions(y_true, y_pred, y_prob, args.variant, output_dir)

    if args.compare:
        plot_comparison_roc(
            pred_path_a=args.compare,
            name_a=args.compare_name,
            pred_path_b=pred_path,
            name_b=args.variant,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()