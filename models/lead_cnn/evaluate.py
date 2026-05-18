"""
Evaluation script for LEAD-CNN.

Produces:
  - Classification report (accuracy, precision, recall, F1 per class + weighted avg)
  - Cohen Kappa score
  - Confusion matrix (printed + saved)
  - Full results saved to results/ as JSON + CSV

Run this after train.py to evaluate the saved best-weights model.
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
)

from dataset import create_generators
from config import MODEL_SAVE_PATH, RESULTS_DIR, CLASS_NAMES
from results_logger import print_final_scores, print_confusion_matrix, save_run_results


def main():
    # ── Load data and model ────
    _, _, test_gen = create_generators()
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)

    # ── Predict ────────
    print("Running predictions on test set...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # Use class names from generator (respects alphabetical folder ordering)
    class_labels = list(test_gen.class_indices.keys())

    # ── Print all results ──────
    print_final_scores(y_true, y_pred, class_labels)
    print_confusion_matrix(y_true, y_pred, class_labels)

    # ── Save results ──────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_run_results(y_true, y_pred, class_labels, RESULTS_DIR)


if __name__ == "__main__":
    main()
