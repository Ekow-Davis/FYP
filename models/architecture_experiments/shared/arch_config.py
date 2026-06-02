"""
Shared configuration for architecture experiments.
Loads base values from lead_cnn/config.py and adds experiment-specific
paths. Each experiment's own config.py imports from here and overrides
what it needs.
"""

import os
import importlib.util

_HERE     = os.path.dirname(os.path.abspath(__file__))   # architecture_experiments/shared/
_MODELS   = os.path.abspath(os.path.join(_HERE, "..", ".."))  # models/
_LEAD_CNN = os.path.join(_MODELS, "lead_cnn")
_ROOT     = os.path.abspath(os.path.join(_MODELS, ".."))      # project root

# Load lead_cnn/config.py by file path to avoid any circular import risk
_spec = importlib.util.spec_from_file_location(
    "lead_cnn_config", os.path.join(_LEAD_CNN, "config.py")
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

# ── Base values inherited from lead_cnn ──────────────────────────────────────
IMG_SIZE      = _base.IMG_SIZE
NUM_CLASSES   = _base.NUM_CLASSES
CLASS_NAMES   = _base.CLASS_NAMES
DATASET_PATH  = _base.DATASET_PATH
RANDOM_SEED   = _base.RANDOM_SEED

# ── Default training hyperparameters (paper optimal) ─────────────────────────
# Each experiment config.py can override any of these.
BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 1e-3

# ── Path helpers ─────────────────────────────────────────────────────────────
ARCH_RESULTS_DIR = os.path.join(_ROOT, "results", "architecture_experiments")
LEAD_CNN_DIR     = _LEAD_CNN
MODELS_DIR       = _MODELS
PROJECT_ROOT     = _ROOT
