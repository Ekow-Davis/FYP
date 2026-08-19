"""
bootstrap_test.py

Bootstrap confidence interval test comparing DSC DimRed vs base LEAD-CNN.

Instead of counting disagreements (like McNemar's), bootstrap resamples
the entire test set thousands of times and measures the accuracy difference
on each resample. This gives a distribution of the difference, from which
we extract a confidence interval.

If the confidence interval excludes zero → the improvement is statistically
real and not due to chance.

Usage:
    python models/architecture_experiments/shared/bootstrap_test.py

Requires that evaluate.py has been run for both models first:
    results/predictions_lead_cnn.npz
    results/architecture_experiments/dsc_dimred/predictions_dsc_dimred.npz
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

# ── File paths ────────────────────────────────────────────────────────────────
LEAD_CNN_PRED   = os.path.join(_ROOT, "results", "predictions_lead_cnn.npz")
DSC_DIMRED_PRED = os.path.join(
    _ROOT, "results", "architecture_experiments",
    "dsc_dimred", "predictions_dsc_dimred.npz"
)
OUTPUT_DIR = os.path.join(_ROOT, "results", "significance_tests")

# ── Bootstrap settings ────────────────────────────────────────────────────────
N_BOOTSTRAP  = 10000   # number of resamples — 10k is standard
CONFIDENCE   = 0.95    # 95% confidence interval
RANDOM_SEED  = 42


def load_predictions(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Predictions not found for {label}:\n  {path}\n"
            f"  Run evaluate.py for that model first."
        )
    data = np.load(path, allow_pickle=True)
    return data['y_true'], data['y_pred']


def bootstrap_accuracy_difference(y_true, y_pred_a, y_pred_b,
                                   n_bootstrap=10000, seed=42):
    """
    Bootstrap resampling to estimate the confidence interval of the
    accuracy difference between model B and model A.

    Returns:
        observed_diff : accuracy(B) - accuracy(A) on the real test set
        ci_lower      : lower bound of the confidence interval
        ci_upper      : upper bound of the confidence interval
        p_value       : proportion of bootstrap samples where diff <= 0
                        (one-tailed: probability that B is not better than A)
        diffs         : full array of bootstrap differences (for plotting)
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)

    correct_a = (y_pred_a == y_true).astype(float)
    correct_b = (y_pred_b == y_true).astype(float)

    # Observed difference on the real test set
    observed_diff = correct_b.mean() - correct_a.mean()

    # Bootstrap
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx          = rng.integers(0, n, size=n)   # resample with replacement
        diffs[i]     = correct_b[idx].mean() - correct_a[idx].mean()

    alpha     = 1 - CONFIDENCE
    ci_lower  = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper  = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    # One-tailed p-value: proportion of bootstrap samples where B is not better
    p_value = float(np.mean(diffs <= 0))

    return observed_diff, ci_lower, ci_upper, p_value, diffs


def print_results(name_a, name_b, acc_a, acc_b,
                  observed_diff, ci_lower, ci_upper, p_value, n):
    sep = "=" * 65
    conf_pct = int(CONFIDENCE * 100)

    print(f"\n{sep}")
    print(f"  Bootstrap Test: {name_a} vs {name_b}")
    print(f"  {N_BOOTSTRAP:,} resamples  |  {conf_pct}% confidence interval")
    print(sep)
    print(f"\n  Accuracy on test set ({n:,} samples):")
    print(f"    {name_a:<20} {acc_a*100:.4f}%")
    print(f"    {name_b:<20} {acc_b*100:.4f}%")
    print(f"    Observed difference : {observed_diff*100:+.4f}% "
          f"(in favour of {name_b})")
    print(f"\n  {conf_pct}% Bootstrap Confidence Interval:")
    print(f"    [{ci_lower*100:+.4f}%,  {ci_upper*100:+.4f}%]")
    print(f"\n  p-value (one-tailed): {p_value:.6f}")
    print(f"    (probability that {name_b} is NOT better than {name_a})")

    print(f"\n  Interpretation:")
    if ci_lower > 0:
        print(f"  ✓ STATISTICALLY SIGNIFICANT")
        print(f"    The entire {conf_pct}% CI is above zero, meaning {name_b}")
        print(f"    outperforms {name_a} with {conf_pct}% confidence.")
        if p_value < 0.01:
            print(f"    p < 0.01: very strong evidence of a real improvement.")
        elif p_value < 0.05:
            print(f"    p < 0.05: strong evidence of a real improvement.")
    elif ci_upper < 0:
        print(f"  ✗ {name_a} significantly outperforms {name_b}.")
    else:
        print(f"  ~ INCONCLUSIVE")
        print(f"    The CI crosses zero — cannot confirm a statistically")
        print(f"    significant difference at the {conf_pct}% level.")
        print(f"    Observed improvement exists but may be within noise.")
    print(sep + "\n")


def plot_bootstrap_distribution(diffs, observed_diff, ci_lower, ci_upper,
                                 name_a, name_b, output_dir):
    """Saves a histogram of the bootstrap difference distribution."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(diffs * 100, bins=80, color='#377eb8', alpha=0.7, edgecolor='white')
    ax.axvline(observed_diff * 100, color='#e41a1c', lw=2.5,
               label=f"Observed diff: {observed_diff*100:+.4f}%")
    ax.axvline(ci_lower * 100, color='black', lw=1.5, linestyle='--',
               label=f"{int(CONFIDENCE*100)}% CI lower: {ci_lower*100:+.4f}%")
    ax.axvline(ci_upper * 100, color='black', lw=1.5, linestyle='-.',
               label=f"{int(CONFIDENCE*100)}% CI upper: {ci_upper*100:+.4f}%")
    ax.axvline(0, color='gray', lw=1, linestyle=':', alpha=0.7,
               label="Zero (no difference)")

    ax.set_xlabel("Accuracy Difference (%): DSC DimRed − LEAD-CNN", fontsize=11)
    ax.set_ylabel("Bootstrap Frequency", fontsize=11)
    ax.set_title(
        f"Bootstrap Distribution of Accuracy Difference\n"
        f"{name_b} vs {name_a}  ({N_BOOTSTRAP:,} resamples)",
        fontsize=12
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"bootstrap_{name_b}_vs_{name_a}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Bootstrap plot saved: {out}")


def main():
    print("\nLoading predictions...")
    y_true_a, y_pred_a = load_predictions(LEAD_CNN_PRED,   "LEAD-CNN")
    y_true_b, y_pred_b = load_predictions(DSC_DIMRED_PRED, "DSC DimRed")

    if len(y_true_a) != len(y_true_b) or not np.array_equal(y_true_a, y_true_b):
        raise ValueError(
            "Test sets don't match. Both models must be evaluated on the "
            "same test set in the same order."
        )

    n     = len(y_true_a)
    acc_a = (y_pred_a == y_true_a).mean()
    acc_b = (y_pred_b == y_true_b).mean()

    print(f"  Samples       : {n:,}")
    print(f"  LEAD-CNN acc  : {acc_a*100:.4f}%")
    print(f"  DSC DimRed acc: {acc_b*100:.4f}%")
    print(f"\nRunning {N_BOOTSTRAP:,} bootstrap resamples...")

    observed_diff, ci_lower, ci_upper, p_value, diffs = bootstrap_accuracy_difference(
        y_true_a, y_pred_a, y_pred_b,
        n_bootstrap=N_BOOTSTRAP,
        seed=RANDOM_SEED
    )

    print_results("LEAD-CNN", "DSC DimRed",
                  acc_a, acc_b,
                  observed_diff, ci_lower, ci_upper, p_value, n)

    # Save results JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp":        timestamp,
        "test":             "bootstrap",
        "model_a":          "LEAD-CNN",
        "model_b":          "DSC DimRed",
        "n_samples":        int(n),
        "n_bootstrap":      N_BOOTSTRAP,
        "confidence":       CONFIDENCE,
        "acc_a":            round(float(acc_a) * 100, 4),
        "acc_b":            round(float(acc_b) * 100, 4),
        "observed_diff":    round(float(observed_diff) * 100, 6),
        "ci_lower":         round(float(ci_lower) * 100, 6),
        "ci_upper":         round(float(ci_upper) * 100, 6),
        "p_value":          round(float(p_value), 6),
        "significant_95":   bool(ci_lower > 0),
    }
    out_path = os.path.join(OUTPUT_DIR,
                            f"bootstrap_leadcnn_vs_dscdimred_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved : {out_path}")

    # Distribution plot
    plot_bootstrap_distribution(
        diffs, observed_diff, ci_lower, ci_upper,
        "LEAD-CNN", "DSC DimRed", OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
