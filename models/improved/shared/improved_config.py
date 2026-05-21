"""
Shared configuration for all improved LEAD-CNN variants.
Imports directly from lead_cnn/config.py by file path to avoid
circular imports (since 'config' as a name is ambiguous when
variant config.py files also exist on sys.path).
"""

import os
import importlib.util

_HERE      = os.path.dirname(os.path.abspath(__file__))          # improved/shared/
_MODELS    = os.path.abspath(os.path.join(_HERE, "..", ".."))    # models/
_LEAD_CNN  = os.path.join(_MODELS, "lead_cnn")
_ROOT      = os.path.abspath(os.path.join(_MODELS, ".."))        # project root

# Load lead_cnn/config.py explicitly by file path — no sys.path ambiguity
_spec = importlib.util.spec_from_file_location(
    "lead_cnn_config",
    os.path.join(_LEAD_CNN, "config.py")
)
_lead_cnn_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lead_cnn_config)

# Re-export everything from lead_cnn config
IMG_SIZE        = _lead_cnn_config.IMG_SIZE
NUM_CLASSES     = _lead_cnn_config.NUM_CLASSES
BATCH_SIZE      = _lead_cnn_config.BATCH_SIZE
EPOCHS          = _lead_cnn_config.EPOCHS
LEARNING_RATE   = _lead_cnn_config.LEARNING_RATE
CLASS_NAMES     = _lead_cnn_config.CLASS_NAMES
RANDOM_SEED     = _lead_cnn_config.RANDOM_SEED
DATASET_PATH    = _lead_cnn_config.DATASET_PATH

# Results root for all improved variants
IMPROVED_RESULTS_DIR = os.path.join(_ROOT, "results", "improved")

# Path helpers other modules can use
LEAD_CNN_DIR = _LEAD_CNN
MODELS_DIR   = _MODELS
PROJECT_ROOT = _ROOT
