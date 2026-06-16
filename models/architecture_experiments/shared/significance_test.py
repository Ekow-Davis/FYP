"""
significance_test.py

Runs McNemar's test to determine whether the performance difference
between DSC DimRed and the base LEAD-CNN is statistically significant.

McNemar's test looks at the DISAGREEMENTS between two models on the
same test set:
  - Cases where LEAD-CNN was correct but DSC DimRed was wrong (b)
  - Cases where DSC DimRed was correct but LEAD-CNN was wrong (c)

If b and c are very unequal, the difference is significant.
A p-value below 0.05 means the improvement is not due to chance.

Usage:
    python models/architecture_experiments/shared/significance_test.py

Requires that evaluate.py has been run for both models first,
which saves predictions_lead_cnn.npz and predictions_dsc_dimred.npz.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

# Prediction file locations
LEAD_CNN_PRED_PATH  = os.path.join(_ROOT, "results", "predictions_lead_cnn.npz")
DSC_DIMRED_PRED_PATH = os.path.join(
    _ROOT, "results", "architecture_experiments", "dsc_dimred",
    "predictions_dsc_dimred.npz"
)
OUTPUT_DIR = os.path.join(_ROOT, "results", "significance_tests")


def load_predictions(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Predictions file not found for {label}:\n  {path}\n"
            f"  Run evaluate.py for that model first."
        )
    data = np.load(path, allow_pickle=True)
    return data['y_true'], data['y_pred']


def mcnemar_test(y_true, y_pred_a, y_pred_b, name_a, name_b):
    """
    Runs McNemar's test comparing model A vs model B.

    Builds a 2×2 contingency table:
        - b: A correct, B wrong
        - c: A wrong,   B correct

    Uses the exact binomial test for small samples (b+c < 25),
    continuity-corrected chi-squared otherwise.
    """
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError:
        print("  statsmodels not installed. Run: pip install statsmodels")
        return None

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table cells
    n_both_correct   = int(np.sum( correct_a &  correct_b))
    n_a_only         = int(np.sum( correct_a & ~correct_b))  # b
    n_b_only         = int(np.sum(~correct_a &  correct_b))  # c
    n_both_wrong     = int(np.sum(~correct_a & ~correct_b))

    table = np.array([[n_both_correct, n_a_only],
                      [n_b_only,       n_both_wrong]])

    # Choose exact test for small discordant pairs
    exact = (n_a_only + n_b_only) < 25
    result = mcnemar(table, exact=exact, correction=not exact)

    return {
        "name_a":           name_a,
        "name_b":           name_b,
        "n_total":          int(len(y_true)),
        "n_both_correct":   n_both_correct,
        "n_a_only":         n_a_only,   # b
        "n_b_only":         n_b_only,   # c
        "n_both_wrong":     n_both_wrong,
        "statistic":        round(float(result.statistic), 6),
        "p_value":          round(float(result.pvalue), 8),
        "significant_005":  bool(result.pvalue < 0.05),
        "significant_001":  bool(result.pvalue < 0.01),
        "test_type":        "exact binomial" if exact else "chi-squared (continuity corrected)",
    }


def print_results(r):
    if r is None:
        return

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  McNemar's Test: {r['name_a']} vs {r['name_b']}")
    print(sep)
    print(f"\n  Contingency Table:")
    print(f"  {'':25} {r['name_b']:>15} correct  {r['name_b']:>15} wrong")
    print(f"  {r['name_a']:>25} correct   {r['n_both_correct']:>15,}  {r['n_a_only']:>15,}")
    print(f"  {r['name_a']:>25} wrong     {r['n_b_only']:>15,}  {r['n_both_wrong']:>15,}")

    print(f"\n  Discordant pairs:")
    print(f"    b ({r['name_a']} correct, {r['name_b']} wrong) : {r['n_a_only']:,}")
    print(f"    c ({r['name_b']} correct, {r['name_a']} wrong) : {r['n_b_only']:,}")

    print(f"\n  Test type  : {r['test_type']}")
    print(f"  Statistic  : {r['statistic']}")
    print(f"  p-value    : {r['p_value']}")

    print(f"\n  Interpretation:")
    if r['significant_001']:
        print(f"  ✓ HIGHLY SIGNIFICANT (p < 0.01)")
        print(f"    The improvement of {r['name_b']} over {r['name_a']} is")
        print(f"    statistically significant and very unlikely due to chance.")
    elif r['significant_005']:
        print(f"  ✓ SIGNIFICANT (p < 0.05)")
        print(f"    The improvement of {r['name_b']} over {r['name_a']} is")
        print(f"    statistically significant.")
    else:
        print(f"  ✗ NOT SIGNIFICANT (p ≥ 0.05)")
        print(f"    The difference between models is within chance variation.")

    print(sep + "\n")


def main():
    print("\nLoading predictions...")

    y_true_a, y_pred_lead = load_predictions(LEAD_CNN_PRED_PATH,  "LEAD-CNN")
    y_true_b, y_pred_dsc  = load_predictions(DSC_DIMRED_PRED_PATH, "DSC DimRed")

    # Sanity check — both models must have been evaluated on the same test set
    if len(y_true_a) != len(y_true_b):
        raise ValueError(
            f"Test set size mismatch: LEAD-CNN={len(y_true_a)}, "
            f"DSC DimRed={len(y_true_b)}. Both must use the same test set."
        )
    if not np.array_equal(y_true_a, y_true_b):
        raise ValueError(
            "True labels don't match between models. "
            "Both must be evaluated on the same test set in the same order."
        )

    print(f"  LEAD-CNN predictions   : {len(y_pred_lead):,} samples")
    print(f"  DSC DimRed predictions : {len(y_pred_dsc):,} samples")
    print(f"  Test set labels match  : ✓")

    # Run McNemar's test — DSC DimRed as model B (the proposed improvement)
    result = mcnemar_test(y_true_a, y_pred_lead, y_pred_dsc,
                          name_a="LEAD-CNN", name_b="DSC DimRed")

    print_results(result)

    # Save results
    if result is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path   = os.path.join(OUTPUT_DIR,
                                  f"mcnemar_leadcnn_vs_dscdimred_{timestamp}.json")
        result["timestamp"] = timestamp
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved: {out_path}\n")


if __name__ == "__main__":
    main()
