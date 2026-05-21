"""
v1_class_weights — Training script.

Identical to lead_cnn/train.py except:
  1. Class weights computed from training set distribution
  2. class_weight dict passed to model.fit()
  3. Results saved to results/improved/v1_class_weights/
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_SHARED   = os.path.join(_HERE, "..", "shared")
_LEAD_CNN = os.path.join(_HERE, "..", "..", "lead_cnn")

sys.path.insert(0, _LEAD_CNN)
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)   # inserted last = highest priority, always wins

# ── Imports ───────────────────────────────────────────────────────────────────
from config import (
    VARIANT_NAME, RESULTS_DIR, MODEL_SAVE_PATH,
    LEARNING_RATE, EPOCHS, RANDOM_SEED,
)
from architecture import build_lead_cnn          # reuse base architecture
from dataset import create_generators            # reuse base data pipeline
from class_weights import get_class_weights
from improved_results_logger import (
    print_model_summary, print_training_history, save_run_results
)


def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_gen, val_gen, _ = create_generators()

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = get_class_weights(train_gen)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_lead_cnn()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    print_model_summary(model, VARIANT_NAME)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
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
            verbose=1,
        ),
    ]

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nTraining {VARIANT_NAME}: {EPOCHS} epochs, "
          f"batch={train_gen.batch_size}, lr={LEARNING_RATE}")
    print("  Class weights active — minority classes penalised more heavily.\n")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,   # ← the only change vs base lead_cnn
    )

    print_training_history(history, VARIANT_NAME)

    # Save history
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f, indent=2
        )

    print(f"Best weights: {MODEL_SAVE_PATH}")
    print(f"History:      {history_path}")

    return history


if __name__ == "__main__":
    main()
