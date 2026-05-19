"""
Shared classifier head for all baseline models.

Architecture (same as LEAD-CNN head for fair comparison):
  GlobalAveragePooling2D
  Dense(128) -> LeakyReLU(0.2) -> BatchNorm -> Dropout(0.25)
  Dense(64)  -> LeakyReLU(0.2) -> BatchNorm -> Dropout(0.5)
  Dense(4)   -> Softmax

Entire backbone is frozen (base/feature-extraction mode, paper Table 5).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensorflow.keras import layers, models
from baseline_config import NUM_CLASSES

LEAKY_ALPHA = 0.2


def build_baseline_model(backbone, model_name):
    """
    Wraps a frozen Keras backbone with the shared classifier head.

    Args:
        backbone:   A tf.keras.Model instance loaded with ImageNet weights.
                    All layers will be frozen.
        model_name: String name for the final model (used in summary).

    Returns:
        Compiled-ready tf.keras.Model
    """

    # Freeze entire backbone
    backbone.trainable = False

    inputs = backbone.input

    # Feature extraction
    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Classifier head — matches LEAD-CNN
    x = layers.Dense(128, name="fc1")(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_ALPHA, name="fc1_act")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(0.25, name="fc1_drop")(x)

    x = layers.Dense(64, name="fc2")(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_ALPHA, name="fc2_act")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Dropout(0.5, name="fc2_drop")(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model
