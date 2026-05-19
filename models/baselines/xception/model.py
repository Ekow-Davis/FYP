import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))

from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from baseline_head import build_baseline_model
from baseline_config import IMG_SIZE

# Xception requires minimum 71x71; paper uses 224x224 which is fine.
# Note: paper Table 3 lists Xception input as 299x299 but Table 5
# evaluates it alongside all others at 224x224. We use 224x224 for consistency.
MODEL_NAME    = "xception"
PREPROCESS_FN = preprocess_input


def build_model():
    backbone = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE,
    )
    return build_baseline_model(backbone, MODEL_NAME)
