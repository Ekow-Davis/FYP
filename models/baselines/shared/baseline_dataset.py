"""
Shared data generators for baseline pretrained models.

Each Keras pretrained model has its own preprocessing function
(e.g. tf.keras.applications.densenet.preprocess_input) which scales
pixels to the range expected by that model. We pass this in as
`preprocess_fn` so the correct normalisation is applied per model.

Augmentation on TRAIN only: horizontal flip, vertical flip, rotation 90°.
Val and test: no augmentation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from baseline_config import IMG_SIZE, BATCH_SIZE, DATASET_PATH, RANDOM_SEED


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

        if augment:
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
