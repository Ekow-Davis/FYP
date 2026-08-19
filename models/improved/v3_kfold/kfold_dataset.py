"""
K-Fold dataset loader for v3_kfold.

Loads all images from train/ and val/ splits into memory as numpy arrays
so sklearn's StratifiedGroupKFold can create fold splits that respect
scan identity.

FIX (leak correction): augmented_data/ contains three physical files per
original scan — the original, a "_rot90" copy, and a "_flip" copy (see
03_data_augmentation.ipynb). The previous version of this loader treated
every file as an independent sample, meaning StratifiedKFold could place
scan42.jpg in one fold's training set and scan42_rot90.jpg in that same
fold's validation set — a form of leak, since both are the same anatomy.

This version now also returns a `groups` array identifying which files
belong to the same original scan (grouped by class + base filename with
the "_rot90"/"_flip" suffix stripped). train.py must use
StratifiedGroupKFold with this groups array so that every copy of a
given scan is always placed in the same fold, never split across
train/validation within a fold.

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

# Suffixes added by 03_data_augmentation.ipynb when pre-generating
# augmented copies. Used to recover the original scan identity.
_AUGMENT_SUFFIXES = ["_rot90", "_flip"]


def _base_scan_id(filename, cls):
    """
    Strips known augmentation suffixes to recover the identity of the
    original scan a file was derived from. Scoped by class to avoid any
    cross-class collision on filename alone.

    e.g. "Tr-me_0674.jpg"       -> "meningioma::Tr-me_0674"
         "Tr-me_0674_rot90.jpg" -> "meningioma::Tr-me_0674"
         "Tr-me_0674_flip.jpg"  -> "meningioma::Tr-me_0674"
    All three resolve to the same group id.
    """
    stem = os.path.splitext(filename)[0]
    for suffix in _AUGMENT_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return f"{cls}::{stem}"


def load_split_to_arrays(split_dir):
    """
    Loads all images from a split directory into numpy arrays, along
    with a groups array identifying which files share the same
    underlying scan (see _base_scan_id).

    Args:
        split_dir: path to e.g. augmented_data/train/

    Returns:
        X:      float32 array (N, H, W, 3), normalised to [0, 1]
        y:      int array (N,), class indices
        groups: str array (N,), scan identity — same value for a scan's
                original/rot90/flip copies
    """
    X, y, groups = [], [], []
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
                groups.append(_base_scan_id(fname, cls))
            except Exception as e:
                print(f"  Warning: skipped {fpath} ({e})")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups, dtype=object)

    n_unique_scans = len(set(groups))
    print(f"  Loaded {len(X)} images ({n_unique_scans} unique scans, "
          f"~{len(X) / max(n_unique_scans, 1):.1f} copies/scan) from {split_dir}")
    return X, y, groups


def load_trainval_arrays():
    """
    Loads train + val splits combined into a single array pair, plus
    the groups array needed for StratifiedGroupKFold.

    Returns:
        X:      (N, H, W, 3)
        y:      (N,) class indices
        groups: (N,) scan identity strings — pass this to
                StratifiedGroupKFold.split(X, y, groups=groups)
    """
    train_dir = os.path.join(DATASET_PATH, "train")
    val_dir   = os.path.join(DATASET_PATH, "val")

    print("\nLoading train + val splits for K-Fold...")
    X_train, y_train, g_train = load_split_to_arrays(train_dir)
    X_val,   y_val,   g_val   = load_split_to_arrays(val_dir)

    X      = np.concatenate([X_train, X_val], axis=0)
    y      = np.concatenate([y_train, y_val], axis=0)
    groups = np.concatenate([g_train, g_val], axis=0)

    n_unique_scans = len(set(groups))
    print(f"  Combined: {len(X)} images across {NUM_CLASSES} classes "
          f"({n_unique_scans} unique underlying scans)\n")
    return X, y, groups


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

    NOTE: since augmented_data/ already contains pre-generated rot90/flip
    copies as physical files, and load_trainval_arrays() loads all of
    them, this function is generally NOT needed on top of that — using
    both means the same rotation/flip transformation is effectively
    applied twice (once by the pre-saved file, once live here). Kept
    for backward compatibility with any script that intentionally wants
    additional live augmentation on top of the pre-augmented set; do not
    call this in new code without checking whether that's actually
    desired.
    """
    datagen = ImageDataGenerator(
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
    )
    gen = datagen.flow(X_batch, y_batch,
                       batch_size=len(X_batch), shuffle=False, seed=seed)
    return next(gen)
