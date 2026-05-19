"""
Shared configuration for all baseline pretrained models.
Hyperparameters from paper Table 3 (pretrained models column):
  Optimizer:    Adam
  LR:           1e-3
  Batch size:   64
  Epochs:       20
  Flip:         horizontal + vertical
  Final layer:  Softmax
"""

import os

IMG_SIZE      = (224, 224, 3)   # all baselines use 224x224
NUM_CLASSES   = 4
BATCH_SIZE    = 64
EPOCHS        = 20
LEARNING_RATE = 1e-3
RANDOM_SEED   = 42

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Paths resolved from this file's location
_HERE  = os.path.dirname(os.path.abspath(__file__))          # baselines/shared/
_ROOT  = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # project root

DATASET_PATH = os.path.join(_ROOT, "data", "augmented_data")
RESULTS_DIR  = os.path.join(_ROOT, "results", "baselines")
