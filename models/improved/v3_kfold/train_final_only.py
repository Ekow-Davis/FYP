"""
train_final_only.py

Runs ONLY the final model training + test evaluation.
Use this if the folds already completed but train.py crashed
during the final model phase (e.g. OOM error).

Fold results in results/improved/v3_kfold/fold_*/ are preserved.
"""

import os
import sys
import json
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

_HERE     = os.path.dirname(os.path.abspath(__file__))
_SHARED   = os.path.join(_HERE, "..", "shared")
_LEAD_CNN = os.path.join(_HERE, "..", "..", "lead_cnn")

sys.path.insert(0, _LEAD_CNN)
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)

from config import (
    VARIANT_NAME, RESULTS_DIR, N_FOLDS,
    LEARNING_RATE, EPOCHS, BATCH_SIZE, RANDOM_SEED, NUM_CLASSES,
)
from architecture import build_lead_cnn
from kfold_dataset import load_trainval_arrays, get_test_generator
from improved_results_logger import (
    print_training_history, print_final_scores,
    print_confusion_matrix, save_run_results
)

# Reduced batch size for final model — it trains on ~19k samples at once
# so we use a smaller batch to reduce peak memory usage
FINAL_BATCH_SIZE = 32


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load fold summary to include in final results
    fold_summary_path = os.path.join(RESULTS_DIR, "kfold_summary.json")
    fold_summary = {}
    if os.path.exists(fold_summary_path):
        with open(fold_summary_path) as f:
            fold_summary = json.load(f)
        print(f"\nLoaded fold summary from: {fold_summary_path}")
        print(f"  Mean accuracy across folds: {fold_summary.get('mean_accuracy', 'N/A'):.4f}")
        print(f"  Std accuracy across folds:  {fold_summary.get('std_accuracy', 'N/A'):.4f}\n")
    else:
        print("Warning: kfold_summary.json not found — fold stats won't be included in results.")

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y = load_trainval_arrays()

    # Explicitly free anything lingering from fold training
    gc.collect()
    tf.keras.backend.clear_session()

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Training final model on all {len(X)} train+val samples...")
    print(f"  Batch size reduced to {FINAL_BATCH_SIZE} to manage memory.")
    print(sep)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    final_model = build_lead_cnn()
    final_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    final_weights_path = os.path.join(RESULTS_DIR, "final_model_best.keras")
    y_cat = to_categorical(y, NUM_CLASSES)

    # 90/10 internal split for callbacks only
    indices     = np.random.permutation(len(X))
    split_idx   = int(0.9 * len(X))
    train_idx_f = indices[:split_idx]
    val_idx_f   = indices[split_idx:]

    final_callbacks = [
        ModelCheckpoint(
            filepath=final_weights_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    final_history = final_model.fit(
        X[train_idx_f], y_cat[train_idx_f],
        validation_data=(X[val_idx_f], y_cat[val_idx_f]),
        epochs=EPOCHS,
        batch_size=FINAL_BATCH_SIZE,
        callbacks=final_callbacks,
        verbose=1,
    )

    print_training_history(final_history, f"{VARIANT_NAME}_final")

    final_history_path = os.path.join(RESULTS_DIR, "final_model_history.json")
    with open(final_history_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals]
             for k, vals in final_history.history.items()},
            f, indent=2
        )

    # Free training data before loading test set
    del X, y, y_cat
    gc.collect()

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\nEvaluating final model on held-out test set...")
    test_gen     = get_test_generator()
    predictions  = final_model.predict(test_gen, verbose=1)
    y_pred       = np.argmax(predictions, axis=1)
    y_true       = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    print_final_scores(y_true, y_pred, class_labels, VARIANT_NAME)
    print_confusion_matrix(y_true, y_pred, class_labels, VARIANT_NAME)

    save_run_results(
        y_true, y_pred, class_labels, VARIANT_NAME, RESULTS_DIR,
        extra_metrics={
            "improvement":    "kfold",
            "n_folds":        N_FOLDS,
            "kfold_mean_acc": fold_summary.get("mean_accuracy"),
            "kfold_std_acc":  fold_summary.get("std_accuracy"),
            "kfold_mean_f1":  fold_summary.get("mean_f1"),
            "kfold_std_f1":   fold_summary.get("std_f1"),
        }
    )

    print(f"  Final model weights: {final_weights_path}\n")


if __name__ == "__main__":
    main()
