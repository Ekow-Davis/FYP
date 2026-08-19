"""
Data generators for LEAD-CNN training.

Paper methodology:
  - Augmentation applied to TRAINING set only (paper Section 3.2, Step 4)
  - Augmentation: horizontal flip, vertical flip, 90 degree rotation
  - Shuffling during all epochs (Table 3)
  - Pixel normalization: rescale 1/255
  - Val and test: normalization only, no augmentation, no shuffle

FIX (double augmentation): DATASET_PATH points at data/augmented_data/,
which 03_data_augmentation.ipynb has ALREADY populated with three
physical files per scan (original, "_rot90", "_flip"). Applying a live
ImageDataGenerator rotation/flip on top of those files meant the same
transformation family was applied twice: once when the file was written
to disk, and again at load time. The train generator below is now
normalisation-only, so augmentation happens exactly once (at the file
level), matching the paper's description.

If you would rather do augmentation live instead of on disk, set
LIVE_AUGMENT = True below AND point DATASET_PATH at data/cleaned_data
in config.py. Do not enable both at once.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMG_SIZE, BATCH_SIZE, DATASET_PATH, RANDOM_SEED

# Augmentation is already baked into the files under augmented_data/,
# so live augmentation is off. See module docstring.
LIVE_AUGMENT = False


def create_generators():
    """
    Returns (train_gen, val_gen, test_gen).

    With LIVE_AUGMENT = False (default), all three generators apply
    normalisation only. Augmentation for the training split comes from
    the pre-generated rot90/flip files already on disk.
    """

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
