"""
v3_kfold — Evaluation script.
Loads the final model (trained on all train+val) and evaluates on test set.
Run this if you want to re-evaluate without re-running all folds.
"""

import os
import sys
import numpy as np
import tensorflow as tf

_HERE     = os.path.dirname(os.path.abspath(__file__))
_SHARED   = os.path.join(_HERE, "..", "shared")
_LEAD_CNN = os.path.join(_HERE, "..", "..", "lead_cnn")

sys.path.insert(0, _LEAD_CNN)
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)   # inserted last = highest priority, always wins

from config import VARIANT_NAME, RESULTS_DIR
from dataset import create_generators
from improved_results_logger import (
    print_final_scores, print_confusion_matrix, save_run_results
)


def main():
    _, _, test_gen = create_generators()

    final_weights = os.path.join(RESULTS_DIR, "final_model_best.keras")
    model = tf.keras.models.load_model(final_weights, compile=False)

    print(f"Evaluating {VARIANT_NAME} final model on test set...")
    predictions  = model.predict(test_gen, verbose=1)
    y_pred       = np.argmax(predictions, axis=1)
    y_true       = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    print_final_scores(y_true, y_pred, class_labels, VARIANT_NAME)
    print_confusion_matrix(y_true, y_pred, class_labels, VARIANT_NAME)
    save_run_results(
        y_true, y_pred, class_labels, VARIANT_NAME, RESULTS_DIR,
        extra_metrics={"improvement": "kfold", "note": "standalone_eval"}
    )


if __name__ == "__main__":
    main()