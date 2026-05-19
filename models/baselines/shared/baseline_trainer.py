"""
Shared training logic for all baseline models.
Each model's train.py calls run_training() with its model and generators.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_config import LEARNING_RATE, EPOCHS, RESULTS_DIR, RANDOM_SEED
from baseline_results_logger import print_model_summary, print_training_history


def run_training(model, train_gen, val_gen, model_name):
    """
    Compiles, trains, and saves the model.

    Args:
        model:      Built (unfrozen head, frozen backbone) Keras model.
        train_gen:  Training data generator.
        val_gen:    Validation data generator.
        model_name: String identifier, e.g. 'densenet201'. Used for file naming.

    Returns:
        history object
    """
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    print_model_summary(model, model_name)

    # Weights saved per model in results/baselines/<model_name>/
    model_results_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    weights_path = os.path.join(model_results_dir, f"{model_name}_best.keras")

    callbacks = [
        ModelCheckpoint(
            filepath=weights_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,        # tighter patience for baselines (20 epoch budget)
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\nTraining {model_name}: {EPOCHS} epochs, "
          f"batch={train_gen.batch_size}, lr={LEARNING_RATE}")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print_training_history(history, model_name)

    # Save training history
    history_path = os.path.join(model_results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f, indent=2
        )

    print(f"Best weights: {weights_path}")
    print(f"History:      {history_path}")

    return history, weights_path
