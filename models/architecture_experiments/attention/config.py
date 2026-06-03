"""
Channel Attention (Squeeze-and-Excitation) experiment — configuration.

All tunable hyperparameters are here. Change a value, rerun train.py.

── What to tune ─────────────────────────────────────────────────────────────
SE_REDUCTION_RATIO : controls how much the SE block compresses channels
    internally before reweighting.
    16  = standard, ~512 extra parameters (recommended starting point)
    8   = more expressive, ~1024 extra parameters
    4   = most expressive, ~2048 extra parameters
    Higher ratio = fewer parameters but less expressive attention.
    Start at 16 — if accuracy plateaus try 8.

LEARNING_RATE : keep at 1e-3 first since backbone is unchanged.
    Only drop to 5e-4 if training is unstable.

Everything else is identical to base LEAD-CNN — this isolates the
effect of the SE block alone.
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

VARIANT_NAME = "attention"

# ── Architecture hyperparameters ──────────────────────────────────────────────
SE_REDUCTION_RATIO = 16     # SE bottleneck ratio (16 = standard, try 8 if needed)
LEAKY_ALPHA        = 0.2    # matches paper Fig.4

# ── Regularisation (identical to base LEAD-CNN) ───────────────────────────────
DROPOUT_CONV = 0.25
DROPOUT_FC1  = 0.25
DROPOUT_FC2  = 0.50

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 1e-3

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR     = os.path.join(ARCH_RESULTS_DIR, VARIANT_NAME)
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "attention_best.keras")
