"""
Shared data generators for architecture experiments.
Identical pipeline to lead_cnn/dataset.py — augmentation on train only.
Imported by each experiment's train.py and evaluate.py.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from arch_config import IMG_SIZE, BATCH_SIZE, DATASET_PATH, RANDOM_SEED


def create_generators(batch_size=None):
    """
    Returns (train_gen, val_gen, test_gen).
    Optionally override batch_size for memory-constrained runs.
    """
    bs = batch_size or BATCH_SIZE

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMG_SIZE[:2],
        batch_size=bs,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED,
    )
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "val"),
        target_size=IMG_SIZE[:2],
        batch_size=bs,
        class_mode='categorical',
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "test"),
        target_size=IMG_SIZE[:2],
        batch_size=bs,
        class_mode='categorical',
        shuffle=False,
    )
    return train_gen, val_gen, test_gen
