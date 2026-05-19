import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))

import numpy as np
import tensorflow as tf
from model import MODEL_NAME, PREPROCESS_FN
from baseline_dataset import create_generators
from baseline_config import RESULTS_DIR
from baseline_results_logger import print_final_scores, print_confusion_matrix, save_run_results


def main():
    _, _, test_gen = create_generators(preprocess_fn=PREPROCESS_FN)

    weights_path = os.path.join(RESULTS_DIR, MODEL_NAME, f"{MODEL_NAME}_best.keras")
    model = tf.keras.models.load_model(weights_path, compile=False)

    print(f"Evaluating {MODEL_NAME} on test set...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    print_final_scores(y_true, y_pred, class_labels, MODEL_NAME)
    print_confusion_matrix(y_true, y_pred, class_labels, MODEL_NAME)
    save_run_results(y_true, y_pred, class_labels, MODEL_NAME)


if __name__ == "__main__":
    main()
