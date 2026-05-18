# Configuration file for LEAD-CNN model
# Hyperparameters match the paper's ablation study optimal values:
#   Batch size:    64    (Table 8 — best result)
#   Learning rate: 1e-3  (Table 9 — best result)
#   Epochs:        50    (Table 10 — best result)

IMG_SIZE = (224, 224, 3)
NUM_CLASSES = 4

# Optimal hyperparameters from paper ablation (Tables 8, 9, 10)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3

# Class names — must match folder names in dataset
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Paths — resolved relative to this config file so they work regardless
# of which directory you run the scripts from
import os
_HERE = os.path.dirname(os.path.abspath(__file__))          # models/lead_cnn/
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))    # project root

DATASET_PATH   = os.path.join(_ROOT, "data", "augmented_data")
MODEL_SAVE_PATH = os.path.join(_HERE, "saved_weights", "lead_cnn_best.keras")
RESULTS_DIR    = os.path.join(_ROOT, "results")

RANDOM_SEED = 42
