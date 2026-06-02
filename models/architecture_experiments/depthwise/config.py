"""
Depthwise Separable Convolution experiment — configuration.

All tunable hyperparameters are here. Change a value, rerun train.py.
Every run saves the config snapshot alongside the results so you can
always trace which settings produced which score.

── What to tune ─────────────────────────────────────────────────────────────
WIDTH_MULTIPLIER : scales the number of filters in every backbone layer.
    1.0 = same filter counts as original LEAD-CNN (32, 32, 64, 64, 128, 128)
    0.75 = 75% of those (24, 24, 48, 48, 96, 96)
    0.5  = 50% — very lightweight, try if accuracy holds up
    Start at 1.0 to isolate the effect of DSC vs standard convolutions.

LEARNING_RATE : DSC models sometimes need a slightly lower LR to converge
    cleanly. Try 1e-3 first (paper optimal), then 5e-4 if unstable.

BATCH_SIZE : keep at 64 (paper optimal) unless you hit memory issues.

DROPOUT_CONV : dropout after each conv block pair. Paper uses 0.25.
    DSC models are sometimes more regularised so you could try 0.2.

DROPOUT_FC1 / DROPOUT_FC2 : classifier head dropout. Paper: 0.25 / 0.5.
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

VARIANT_NAME = "depthwise"

# ── Architecture hyperparameters ──────────────────────────────────────────────
WIDTH_MULTIPLIER = 1.0    # scale backbone filter counts (1.0 = same as LEAD-CNN)
LEAKY_ALPHA      = 0.2    # LeakyReLU slope (matches Fig.4 of paper)

# ── Regularisation ────────────────────────────────────────────────────────────
DROPOUT_CONV = 0.25       # after each backbone conv block pair
DROPOUT_FC1  = 0.25       # after Dense(128)
DROPOUT_FC2  = 0.50       # after Dense(64)

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 5e-4

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR     = os.path.join(ARCH_RESULTS_DIR, VARIANT_NAME)
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "depthwise_best.keras")
