"""
Combined experiment (DSC + SE) — configuration.

Both depthwise separable convolutions AND channel attention together.
Tune this AFTER you have good individual results from depthwise/ and attention/.
The optimal settings here will likely be close to whichever individual
experiment performed best, but may need slight adjustment.

── What to tune ─────────────────────────────────────────────────────────────
WIDTH_MULTIPLIER   : same as depthwise/ — start at whatever worked there
SE_REDUCTION_RATIO : same as attention/ — start at whatever worked there
LEARNING_RATE      : DSC is sensitive to LR, start at 5e-4 if 1e-3 was
                     unstable in the depthwise experiment
"""

import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.join(_HERE, "..", "shared")
sys.path.insert(0, _SHARED)

from arch_config import (
    IMG_SIZE, NUM_CLASSES, CLASS_NAMES, DATASET_PATH,
    RANDOM_SEED, ARCH_RESULTS_DIR, LEAD_CNN_DIR, PROJECT_ROOT,
)

VARIANT_NAME = "combined"

# ── Architecture hyperparameters ──────────────────────────────────────────────
WIDTH_MULTIPLIER   = 1.0    # update after depthwise results
SE_REDUCTION_RATIO = 4     # update after attention results
LEAKY_ALPHA        = 0.2

# ── Regularisation ────────────────────────────────────────────────────────────
DROPOUT_CONV = 0.25
DROPOUT_FC1  = 0.25
DROPOUT_FC2  = 0.50

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 5e-4    # start lower since DSC needs more careful tuning

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR     = os.path.join(ARCH_RESULTS_DIR, VARIANT_NAME)
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "combined_best.keras")
