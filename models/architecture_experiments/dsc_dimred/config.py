"""
DSC Dimension Reduction experiment — configuration.

This experiment keeps the backbone EXACTLY as the original LEAD-CNN paper,
but replaces the 3x3 and 5x5 convolutions inside the Modified Dimension
Reduction Block with depthwise separable equivalents.

The 1x1 convolutions in the block are kept standard since they are already
parameter-efficient and serve as channel mixers (depthwise can't do that).

── What to tune ─────────────────────────────────────────────────────────────
LEARNING_RATE : start at 1e-3 (paper optimal for backbone)
    Only reduce if val loss is unstable.

DSC_USE_BN : whether to add BatchNorm after DSC layers in the block.
    True = recommended based on depthwise experiment findings.
    False = matches original block style (no BN).

Everything else identical to base LEAD-CNN.
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

VARIANT_NAME = "dsc_dimred"

# ── Architecture hyperparameters ──────────────────────────────────────────────
DSC_USE_BN  = True      # add BatchNorm after DSC layers in dim reduction block
LEAKY_ALPHA = 0.2       # matches paper Fig.4

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
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "dsc_dimred_best.keras")
