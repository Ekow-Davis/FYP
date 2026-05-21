"""
v4_combined configuration.
Combines: K-Fold cross validation + Class weights.
Grad-CAM runs separately on the saved final model weights.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "shared"))

from improved_config import *

VARIANT_NAME = "v4_combined"
RESULTS_DIR  = os.path.join(IMPROVED_RESULTS_DIR, VARIANT_NAME)
N_FOLDS      = 5
