"""
K-Fold dataset loader for v3_kfold.

Loads all images from train/ and val/ splits into memory as numpy arrays
so sklearn's KFold can create arbitrary fold splits.

Test set is never touched here — it stays as a DirectoryIterator
for final evaluation only.
"""

import os
import sys
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

_HERE     = os.path.dirname(os.path.abspath(__file__))
_LEAD_CNN = os.path.join(_HERE, "..", "..", "lead_cnn")
sys.path.insert(0, _LEAD_CNN)
sys.path.insert(0, _HERE)   # highest priority

from config import IMG_SIZE, NUM_CLASSES, DATASET_PATH, RANDOM_SEED, CLASS_NAMES


def load_split_to_arrays(split_dir):
    """
    Loads all images from a split directory into numpy arrays.

    Args:
        split_dir: path to e.g. augmented_data/train/

    Returns:
        X: float32 array (N, H, W, 3), normalised to [0, 1]
        y: int array (N,), class indices
    """
    X, y = [], []
    class_dirs = sorted(os.listdir(split_dir))

    # Build class-to-index mapping consistent with Keras alphabetical ordering
    class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}

    for cls in class_dirs:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        idx = class_to_idx[cls]
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            fpath = os.path.join(cls_dir, fname)
            try:
                img = load_img(fpath, target_size=IMG_SIZE[:2])
                arr = img_to_array(img) / 255.0
                X.append(arr)
                y.append(idx)
            except Exception as e:
                print(f"  Warning: skipped {fpath} ({e})")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"  Loaded {len(X)} images from {split_dir}")
    return X, y


def load_trainval_arrays():
    """
    Loads train + val splits combined into a single array pair.
    This is what KFold will split into folds.

    Returns:
        X: (N, H, W, 3)
        y: (N,) class indices
    """
    train_dir = os.path.join(DATASET_PATH, "train")
    val_dir   = os.path.join(DATASET_PATH, "val")

    print("\nLoading train + val splits for K-Fold...")
    X_train, y_train = load_split_to_arrays(train_dir)
    X_val,   y_val   = load_split_to_arrays(val_dir)

    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)

    print(f"  Combined: {len(X)} images across {NUM_CLASSES} classes\n")
    return X, y


def get_test_generator():
    """
    Returns a test generator (no augmentation) for final holdout evaluation.
    Same as base lead_cnn dataset.py but imported here for convenience.
    """
    from dataset import create_generators
    _, _, test_gen = create_generators()
    return test_gen


def augment_batch(X_batch, y_batch, seed=RANDOM_SEED):
    """
    Applies training augmentation to a numpy batch.
    Used inside the fold training loop.
    """
    datagen = ImageDataGenerator(
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
    )
    # flow returns an iterator; we take one batch of the same size
    gen = datagen.flow(X_batch, y_batch,
                       batch_size=len(X_batch), shuffle=False, seed=seed)
    return next(gen)
