"""
Evaluation script for LEAD-CNN.
Runs metrics, ROC/AUC curves, and saves raw predictions for McNemar's test.
"""

import os
import sys
import numpy as np
import tensorflow as tf

_HERE        = os.path.dirname(os.path.abspath(__file__))
_ARCH_SHARED = os.path.join(_HERE, "..", "architecture_experiments", "shared")

sys.path.insert(0, _HERE)
sys.path.insert(0, _ARCH_SHARED)

from config import MODEL_SAVE_PATH, RESULTS_DIR, CLASS_NAMES
from dataset import create_generators
from results_logger import (
    print_final_scores, print_confusion_matrix, save_run_results
)
from arch_results_logger import plot_roc_curves, save_predictions

VARIANT_NAME = "lead_cnn"


def main():
    _, _, test_gen = create_generators()

    model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)

    print(f"Evaluating LEAD-CNN on test set...")
    y_prob       = model.predict(test_gen, verbose=1)
    y_pred       = np.argmax(y_prob, axis=1)
    y_true       = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # lead_cnn results_logger takes 3 args (no variant_name parameter)
    print_final_scores(y_true, y_pred, class_labels)
    print_confusion_matrix(y_true, y_pred, class_labels)
    save_run_results(y_true, y_pred, class_labels, RESULTS_DIR)

    # ROC curves and AUC scores — uses arch_results_logger which takes variant_name
    plot_roc_curves(y_true, y_prob, class_labels,
                    VARIANT_NAME, RESULTS_DIR)

    # Save raw predictions so McNemar's test can compare against DSC DimRed
    save_predictions(y_true, y_pred, y_prob,
                     VARIANT_NAME, RESULTS_DIR)


if __name__ == "__main__":
    main()