"""
v3_kfold — Training script.

Runs 5-fold cross validation on the combined train+val split.
For each fold:
  - Trains a fresh LEAD-CNN from scratch
  - Evaluates on the held-out fold
  - Saves fold weights and metrics

After all folds:
  - Reports mean ± std across folds (variance analysis)
  - Trains a final model on ALL train+val data
  - Evaluates once on the untouched test set (final holdout score)
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score
)

_HERE     = os.path.dirname(os.path.abspath(__file__))
_SHARED   = os.path.join(_HERE, "..", "shared")
_LEAD_CNN = os.path.join(_HERE, "..", "..", "lead_cnn")

sys.path.insert(0, _LEAD_CNN)
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)   # inserted last = highest priority, always wins

from config import (
    VARIANT_NAME, RESULTS_DIR, N_FOLDS,
    LEARNING_RATE, EPOCHS, BATCH_SIZE, RANDOM_SEED, NUM_CLASSES,
)
from architecture import build_lead_cnn
from kfold_dataset import load_trainval_arrays, get_test_generator
from improved_results_logger import (
    print_model_summary, print_training_history,
    print_kfold_summary, print_final_scores,
    print_confusion_matrix, save_run_results
)


def train_fold(X_train, y_train, X_val, y_val, fold_num, fold_dir):
    """Trains one fold and returns history + predictions on the val fold."""
    tf.random.set_seed(RANDOM_SEED + fold_num)
    np.random.seed(RANDOM_SEED + fold_num)

    model = build_lead_cnn()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    os.makedirs(fold_dir, exist_ok=True)
    weights_path = os.path.join(fold_dir, f"fold{fold_num}_best.keras")

    callbacks = [
        ModelCheckpoint(
            filepath=weights_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0,
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

    # Convert labels to categorical for training
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat   = to_categorical(y_val,   NUM_CLASSES)

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on val fold
    preds  = model.predict(X_val, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    fold_metrics = {
        "fold":        fold_num,
        "accuracy":    accuracy_score(y_val, y_pred) * 100,
        "precision":   precision_score(y_val, y_pred, average='weighted') * 100,
        "recall":      recall_score(y_val, y_pred, average='weighted') * 100,
        "f1":          f1_score(y_val, y_pred, average='weighted') * 100,
        "cohen_kappa": cohen_kappa_score(y_val, y_pred),
        "best_val_acc": max(history.history['val_accuracy']),
        "epochs_run":   len(history.history['accuracy']),
    }

    # Save fold history
    history_path = os.path.join(fold_dir, f"fold{fold_num}_history.json")
    with open(history_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f, indent=2
        )

    return fold_metrics, history


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y = load_trainval_arrays()

    # ── K-Fold cross validation ───────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    sep = "=" * 68

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n{sep}")
        print(f"  FOLD {fold_num} / {N_FOLDS}")
        print(f"  Train: {len(train_idx)} samples  |  Val: {len(val_idx)} samples")
        print(sep)

        X_train_f, y_train_f = X[train_idx], y[train_idx]
        X_val_f,   y_val_f   = X[val_idx],   y[val_idx]

        fold_dir     = os.path.join(RESULTS_DIR, f"fold_{fold_num}")
        metrics, history = train_fold(
            X_train_f, y_train_f, X_val_f, y_val_f, fold_num, fold_dir
        )
        fold_results.append(metrics)

        print(f"\n  Fold {fold_num} results:  "
              f"Acc={metrics['accuracy']:.2f}%  "
              f"F1={metrics['f1']:.2f}%  "
              f"Kappa={metrics['cohen_kappa']:.4f}  "
              f"(ran {metrics['epochs_run']} epochs)")

    # ── K-Fold summary ────────────────────────────────────────────────────────
    print_kfold_summary(fold_results, VARIANT_NAME)

    fold_summary_path = os.path.join(RESULTS_DIR, "kfold_summary.json")
    accs   = [r['accuracy']    for r in fold_results]
    f1s    = [r['f1']          for r in fold_results]
    kappas = [r['cohen_kappa'] for r in fold_results]

    with open(fold_summary_path, "w") as f:
        json.dump({
            "folds":         fold_results,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1":       float(np.mean(f1s)),
            "std_f1":        float(np.std(f1s)),
            "mean_kappa":    float(np.mean(kappas)),
            "std_kappa":     float(np.std(kappas)),
        }, f, indent=2)
    print(f"  K-Fold summary saved: {fold_summary_path}")

    # ── Final model: train on ALL train+val, evaluate on test ─────────────────
    print(f"\n{sep}")
    print(f"  Training final model on all {len(X)} train+val samples...")
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

    # For the final model we use a 90/10 internal val split just for callbacks
    split_idx   = int(0.9 * len(X))
    indices     = np.random.permutation(len(X))
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
        batch_size=BATCH_SIZE,
        callbacks=final_callbacks,
        verbose=1,
    )

    print_training_history(final_history, f"{VARIANT_NAME}_final")

    # Save final history
    final_history_path = os.path.join(RESULTS_DIR, "final_model_history.json")
    with open(final_history_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals]
             for k, vals in final_history.history.items()},
            f, indent=2
        )

    # ── Test set evaluation ───────────────────────────────────────────────────
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
            "kfold_mean_acc": round(float(np.mean(accs)), 4),
            "kfold_std_acc":  round(float(np.std(accs)),  4),
            "kfold_mean_f1":  round(float(np.mean(f1s)),  4),
            "kfold_std_f1":   round(float(np.std(f1s)),   4),
        }
    )

    print(f"  Final model weights: {final_weights_path}\n")


if __name__ == "__main__":
    main()
