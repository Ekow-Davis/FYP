"""
Shared data generators for baseline pretrained models.

Each Keras pretrained model has its own preprocessing function
(e.g. tf.keras.applications.densenet.preprocess_input) which scales
pixels to the range expected by that model. We pass this in as
`preprocess_fn` so the correct normalisation is applied per model.

FIX (double augmentation): DATASET_PATH points at data/augmented_data/,
which already contains three physical files per scan (original,
"_rot90", "_flip"). Live ImageDataGenerator augmentation on top of that
applied the same transformation family twice. Augmentation is now
file-level only, so the train generator applies preprocessing only.
See lead_cnn/dataset.py for the same change.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from baseline_config import IMG_SIZE, BATCH_SIZE, DATASET_PATH, RANDOM_SEED

# Augmentation already baked into the files under augmented_data/.
LIVE_AUGMENT = False


def create_generators(preprocess_fn=None):
    """
    Args:
        preprocess_fn: model-specific preprocessing function from
                       tf.keras.applications.<model>.preprocess_input.
                       If None, falls back to rescale 1/255.

    Returns:
        (train_gen, val_gen, test_gen)
    """

    def make_datagen(augment=False):
        kwargs = {}
        if preprocess_fn is not None:
            kwargs['preprocessing_function'] = preprocess_fn
        else:
            kwargs['rescale'] = 1.0 / 255

        # Only applied when LIVE_AUGMENT is explicitly enabled.
        if augment and LIVE_AUGMENT:
            kwargs['rotation_range']   = 90
            kwargs['horizontal_flip']  = True
            kwargs['vertical_flip']    = True

        return ImageDataGenerator(**kwargs)

    train_datagen = make_datagen(augment=True)
    eval_datagen  = make_datagen(augment=False)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_gen = eval_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "val"),
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "test"),
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    return train_gen, val_gen, test_gen
