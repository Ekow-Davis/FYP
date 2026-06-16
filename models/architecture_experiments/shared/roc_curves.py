"""
roc_curves.py — Standalone ROC/AUC generator.

Loads any saved .keras model, runs it on the test set,
and produces ROC curves + AUC scores.

Usage:
    # DSC DimRed
    python models/architecture_experiments/shared/roc_curves.py \
        --model results/architecture_experiments/dsc_dimred/dsc_dimred_best.keras \
        --variant dsc_dimred

    # Base LEAD-CNN
    python models/architecture_experiments/shared/roc_curves.py \
        --model models/lead_cnn/saved_weights/lead_cnn_best.keras \
        --variant lead_cnn

    # Both on one plot for comparison
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

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

sys.path.insert(0, _HERE)

from arch_config  import DATASET_PATH, IMG_SIZE, RANDOM_SEED
from arch_dataset import create_generators
from arch_results_logger import plot_roc_curves, save_predictions


def plot_comparison_roc(pred_path_a, name_a, pred_path_b, name_b, output_dir):
    """
    Plots macro-average ROC curves for two models on the same axes
    for a direct visual comparison.
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

    def load_and_compute(path):
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
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= n_cls
        macro_auc = auc(all_fpr, mean_tpr)
        return all_fpr, mean_tpr, macro_auc

    fpr_a, tpr_a, auc_a = load_and_compute(pred_path_a)
    fpr_b, tpr_b, auc_b = load_and_compute(pred_path_b)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_a, tpr_a, color='#377eb8', lw=2,
            label=f"{name_a} — Macro AUC = {auc_a:.4f}")
    ax.plot(fpr_b, tpr_b, color='#e41a1c', lw=2, linestyle='--',
            label=f"{name_b} — Macro AUC = {auc_b:.4f}")
    ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Macro-Average ROC Comparison', fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"roc_comparison_{name_a}_vs_{name_b}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Comparison ROC saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ROC/AUC curves for a saved LEAD-CNN variant."
    )
    parser.add_argument("--model",    required=True,
                        help="Path to .keras model file.")
    parser.add_argument("--variant",  required=True,
                        help="Name for this model (used in titles and filenames).")
    parser.add_argument("--output",   default=None,
                        help="Output directory. Defaults to results/roc/")
    parser.add_argument("--compare",  default=None,
                        help="Path to a .npz predictions file for comparison overlay.")
    parser.add_argument("--compare-name", default="comparison",
                        help="Name for the comparison model.")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(_ROOT, "results", "roc")
    os.makedirs(output_dir, exist_ok=True)

    # Load data and model
    _, _, test_gen = create_generators()
    model = tf.keras.models.load_model(args.model, compile=False)

    print(f"\nRunning predictions for {args.variant}...")
    y_prob       = model.predict(test_gen, verbose=1)
    y_pred       = np.argmax(y_prob, axis=1)
    y_true       = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # Per-class ROC curves
    plot_roc_curves(y_true, y_prob, class_labels, args.variant, output_dir)

    # Save predictions
    pred_path = save_predictions(y_true, y_pred, y_prob, args.variant, output_dir)

    # Optional comparison overlay
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
