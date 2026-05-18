"""
Training script for LEAD-CNN.

Uses paper's optimal hyperparameters:
  - Adam optimizer, lr=1e-3
  - Batch size 64
  - 50 epochs
  - ModelCheckpoint (saves best val_accuracy weights)
  - EarlyStopping (patience=10, monitors val_loss)
  - Training history saved to results/
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from architecture import build_lead_cnn
from dataset import create_generators
from config import LEARNING_RATE, EPOCHS, MODEL_SAVE_PATH, RESULTS_DIR, RANDOM_SEED
from results_logger import print_model_summary, print_training_history, save_run_results


def main():
    # ── Reproducibility ───────────────
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Data ───────────────
    train_gen, val_gen, _ = create_generators()

    # ── Model ────────────────────
    model = build_lead_cnn()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Print full parameter table before training
    print_model_summary(model)

    # ── Callbacks ───────────────
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    callbacks = [
        # Save best weights by validation accuracy
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        # Stop early if val_loss stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR on plateau (helps avoid local minima near convergence)
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── Training ─────────────────
    print(f"\nStarting training: {EPOCHS} epochs, batch={train_gen.batch_size}, lr={LEARNING_RATE}")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # ── Print epoch-by-epoch summary ─────────────
    print_training_history(history)

    # ── Save run results ───────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    print(f"Best model weights saved to: {MODEL_SAVE_PATH}")

    return history


if __name__ == "__main__":
    main()
