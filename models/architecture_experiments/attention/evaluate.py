"""
Attention (SE) experiment — evaluation script.
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
    "attention_config", os.path.join(_HERE, "config.py")
)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

from arch_dataset import create_generators
from arch_results_logger import (
    print_final_scores, print_confusion_matrix, save_run_results
)


def main():
    _, _, test_gen = create_generators(batch_size=config.BATCH_SIZE)

    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH, compile=False)

    print(f"\nEvaluating {config.VARIANT_NAME} on test set...")
    predictions  = model.predict(test_gen, verbose=1)
    y_pred       = np.argmax(predictions, axis=1)
    y_true       = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    print_final_scores(y_true, y_pred, class_labels, config.VARIANT_NAME,
                       config=config)
    print_confusion_matrix(y_true, y_pred, class_labels, config.VARIANT_NAME)
    save_run_results(
        y_true, y_pred, class_labels, config.VARIANT_NAME,
        config.RESULTS_DIR, config=config,
        extra_metrics={"architecture": "se_channel_attention"}
    )


if __name__ == "__main__":
    main()
