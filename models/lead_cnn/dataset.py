"""
Data generators for LEAD-CNN training.

Paper methodology:
  - Augmentation applied to TRAINING set only (paper Section 3.2, Step 4)
  - Augmentation: horizontal flip, vertical flip, 90° rotation
  - Shuffling during all epochs (Table 3)
  - Pixel normalization: rescale 1/255
  - Val and test: normalization only, no augmentation, no shuffle
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMG_SIZE, BATCH_SIZE, DATASET_PATH, RANDOM_SEED


def create_generators():
    """
    Returns (train_gen, val_gen, test_gen).

    Train generator applies augmentation; val/test do not.
    """

    # Training generator — augmentation matches paper (Step 4):
    #   horizontal flip, vertical flip, 90° rotation (via rotation_range=90)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=90,          # enables 90° rotations
        horizontal_flip=True,
        vertical_flip=True,
    )

    # Val/Test generators — normalisation only
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        f"{DATASET_PATH}/train",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_gen = eval_datagen.flow_from_directory(
        f"{DATASET_PATH}/val",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        f"{DATASET_PATH}/test",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,  # must be False for evaluate.py predictions to align
    )

    return train_gen, val_gen, test_gen
