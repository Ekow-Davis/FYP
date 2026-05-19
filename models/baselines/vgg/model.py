import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from baseline_head import build_baseline_model
from baseline_config import IMG_SIZE

MODEL_NAME    = "vgg19"
PREPROCESS_FN = preprocess_input


def build_model():
    backbone = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE,
    )
    return build_baseline_model(backbone, MODEL_NAME)
