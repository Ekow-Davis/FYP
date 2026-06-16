"""
DSC DimRed experiment — evaluation script.
Runs metrics, ROC/AUC, and saves predictions for McNemar's test.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import importlib.util

_HERE   = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.join(_HERE, "..", "shared")
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)

_spec = importlib.util.spec_from_file_location(
    "dsc_dimred_config", os.path.join(_HERE, "config.py")
)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

from arch_dataset import create_generators
from arch_results_logger import (
    print_final_scores, print_confusion_matrix, save_run_results,
    plot_roc_curves, save_predictions
)


def main():
    _, _, test_gen = create_generators(batch_size=config.BATCH_SIZE)

    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH, compile=False)

    print(f"\nEvaluating {config.VARIANT_NAME} on test set...")
    y_prob       = model.predict(test_gen, verbose=1)
    y_pred       = np.argmax(y_prob, axis=1)
    y_true       = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    print_final_scores(y_true, y_pred, class_labels, config.VARIANT_NAME,
                       config=config)
    print_confusion_matrix(y_true, y_pred, class_labels, config.VARIANT_NAME)
    save_run_results(
        y_true, y_pred, class_labels, config.VARIANT_NAME,
        config.RESULTS_DIR, config=config,
        extra_metrics={"architecture": "dsc_in_dimred_block_only"}
    )

    # ROC curves and AUC scores
    plot_roc_curves(y_true, y_prob, class_labels,
                    config.VARIANT_NAME, config.RESULTS_DIR)

    # Save raw predictions for McNemar's test
    save_predictions(y_true, y_pred, y_prob,
                     config.VARIANT_NAME, config.RESULTS_DIR)


if __name__ == "__main__":
    main()
