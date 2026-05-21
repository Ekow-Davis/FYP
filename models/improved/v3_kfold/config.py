"""
v3_kfold configuration.
K-Fold is run on train+val data only. Test set remains untouched as final holdout.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "shared"))

from improved_config import *

VARIANT_NAME = "v3_kfold"
RESULTS_DIR  = os.path.join(IMPROVED_RESULTS_DIR, VARIANT_NAME)

N_FOLDS      = 5       # standard k-fold value
# Each fold trains for EPOCHS (50) with early stopping,
# so worst case is 5 * 50 = 250 epochs total — early stopping
# will cut this significantly in practice.
