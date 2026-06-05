"""
DSC DimRed + SE Attention experiment — configuration.

This is the final combined architectural experiment:
  - Backbone: UNCHANGED (standard Conv2D, identical to paper)
  - Dim Reduction Block: DSC replaces 3x3 and 5x5 branches
  - SE attention: inserted after concatenation
  - Classifier head: UNCHANGED

Start with the best settings found in each individual experiment:
  DSC_USE_BN         = True  (from dsc_dimred findings)
  SE_REDUCTION_RATIO = 4     (best from attention experiment)
  LEARNING_RATE      = 1e-3  (paper optimal, backbone unchanged)
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

VARIANT_NAME = "dsc_dimred_attention"

# ── Architecture hyperparameters ──────────────────────────────────────────────
DSC_USE_BN         = True   # BatchNorm after DSC layers (from dsc_dimred findings)
SE_REDUCTION_RATIO = 4      # best ratio from attention experiment
LEAKY_ALPHA        = 0.2

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
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "dsc_dimred_attention_best.keras")
