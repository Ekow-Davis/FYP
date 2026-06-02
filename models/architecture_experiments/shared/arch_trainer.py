"""
Shared training logic for all architecture experiments.
Each experiment's train.py calls run_training() with its built model.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from arch_results_logger import print_model_summary, print_training_history


def run_training(model, train_gen, val_gen, config):
    """
    Compiles, trains, and saves the model.

    Args:
        model:     Built Keras model (uncompiled)
        train_gen: Training data generator
        val_gen:   Validation data generator
        config:    The experiment's config module (has VARIANT_NAME,
                   RESULTS_DIR, MODEL_SAVE_PATH, LEARNING_RATE,
                   EPOCHS, RANDOM_SEED)

    Returns:
        history object
    """
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    print_model_summary(model, config.VARIANT_NAME)
    print(f"\n  Hyperparameters:")
    print(f"    Learning rate : {config.LEARNING_RATE}")
    print(f"    Batch size    : {train_gen.batch_size}")
    print(f"    Epochs        : {config.EPOCHS}")
    print(f"    Early stop    : patience=10 (val_loss)")
    print(f"    LR reduction  : factor=0.5, patience=5\n")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
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

    print(f"Training {config.VARIANT_NAME}...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
    )

    print_training_history(history, config.VARIANT_NAME)

    history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals]
             for k, vals in history.history.items()},
            f, indent=2
        )

    print(f"  Best weights : {config.MODEL_SAVE_PATH}")
    print(f"  History      : {history_path}\n")
    return history
