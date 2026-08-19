"""
Shared data generators for architecture experiments.
Identical pipeline to lead_cnn/dataset.py.
Imported by each experiment's train.py and evaluate.py.

FIX (double augmentation): DATASET_PATH points at data/augmented_data/,
which already contains three physical files per scan (original,
"_rot90", "_flip") written by 03_data_augmentation.ipynb. Live
ImageDataGenerator augmentation on top of those files applied the same
transformation family twice. The train generator is now
normalisation-only so augmentation happens exactly once, at the file
level. See lead_cnn/dataset.py for the same change and the LIVE_AUGMENT
escape hatch.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from arch_config import IMG_SIZE, BATCH_SIZE, DATASET_PATH, RANDOM_SEED

# Augmentation already baked into the files under augmented_data/.
LIVE_AUGMENT = False


def create_generators(batch_size=None):
    """
    Returns (train_gen, val_gen, test_gen).
    Optionally override batch_size for memory-constrained runs.
    """
    bs = batch_size or BATCH_SIZE

    train_kwargs = {"rescale": 1.0 / 255}
    if LIVE_AUGMENT:
        train_kwargs.update({
            "rotation_range":  90,
            "horizontal_flip": True,
            "vertical_flip":   True,
        })

    train_datagen = ImageDataGenerator(**train_kwargs)
    eval_datagen  = ImageDataGenerator(rescale=1.0 / 255)

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
