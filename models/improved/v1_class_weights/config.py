"""
v1_class_weights configuration.
Inherits everything from improved/shared/improved_config.py.
Only adds the variant name and results path.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "shared"))

from improved_config import *   # pulls in all base hyperparameters

VARIANT_NAME = "v1_class_weights"
RESULTS_DIR  = os.path.join(IMPROVED_RESULTS_DIR, VARIANT_NAME)
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "v1_best.keras")
