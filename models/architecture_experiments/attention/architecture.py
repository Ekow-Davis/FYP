"""
SE-LEAD-CNN: LEAD-CNN with Squeeze-and-Excitation channel attention.

Changes vs original LEAD-CNN:
  - Backbone: UNCHANGED (same 6 standard Conv2D layers)
  - Dimension Reduction Block: UNCHANGED
  - SE block inserted AFTER the dimension reduction block concatenation
    and BEFORE the flatten layer
  - Classifier head: UNCHANGED

What the SE block does:
  1. Squeeze: GlobalAveragePooling collapses 10x10x64 → 64 values,
     one per channel, representing global channel-wise statistics
  2. Excitation: Two Dense layers (64→4→64) learn a reweighting vector
     — which of the 64 channels matter most for this input
  3. Scale: The input feature map is multiplied channel-wise by the
     learned weights — important channels amplified, weak ones suppressed

Why after the dim reduction block:
  The concatenation merges 4 branches (1x1, 3x3, 5x5, MaxPool+1x1).
  Different tumour types benefit from different scale features — pituitary
  from fine 1x1 features, glioma from coarser 5x5 features. SE lets the
  model learn this weighting dynamically per input rather than treating
  all branches equally.

Parameter cost:
  SE adds 2 * C * C/r parameters where C=64, r=SE_REDUCTION_RATIO
  At r=16: 2 * 64 * 4 = 512 extra parameters — negligible.
"""

import sys
import os
import importlib.util

from tensorflow.keras import layers, models

_HERE   = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.join(_HERE, "..", "shared")
sys.path.insert(0, _SHARED)

from arch_config import IMG_SIZE, NUM_CLASSES

_cfg_spec = importlib.util.spec_from_file_location(
    "attention_config", os.path.join(_HERE, "config.py")
)
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)


def squeeze_and_excitation(x, ratio, name="se"):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).

    Args:
        x:     Input tensor, shape (batch, H, W, C)
        ratio: Reduction ratio for the bottleneck Dense layer
        name:  Layer name prefix

    Returns:
        Rescaled tensor, same shape as input
    """
    channels = x.shape[-1]

    # Squeeze: global average pool → (batch, C)
    se = layers.GlobalAveragePooling2D(name=f"{name}_squeeze")(x)

    # Excitation: FC → ReLU → FC → Sigmoid → (batch, C)
    se = layers.Dense(max(1, channels // ratio),
                      activation='relu', name=f"{name}_fc1")(se)
    se = layers.Dense(channels,
                      activation='sigmoid', name=f"{name}_fc2")(se)

    # Reshape to (batch, 1, 1, C) for broadcasting
    se = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(se)

    # Scale: multiply input feature map by learned channel weights
    return layers.Multiply(name=f"{name}_scale")([x, se])


def dimension_reduction_block(x, name="dim_red"):
    """
    Modified Dimension Reduction Block — UNCHANGED from paper (Fig.4).
    """
    a = layers.Conv2D(16, (1,1), padding='same', name=f"{name}_a_conv1x1")(x)
    a = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_a_act1")(a)
    a = layers.Conv2D(16, (3,3), padding='same', name=f"{name}_a_conv3x3")(a)
    a = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_a_act2")(a)

    b = layers.Conv2D(16, (1,1), padding='same', name=f"{name}_b_conv1x1")(x)
    b = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_b_act1")(b)
    b = layers.Conv2D(16, (5,5), padding='same', name=f"{name}_b_conv5x5")(b)
    b = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_b_act2")(b)

    c = layers.MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same',
                            name=f"{name}_c_pool")(x)
    c = layers.Conv2D(16, (1,1), padding='same', name=f"{name}_c_conv1x1")(c)
    c = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_c_act")(c)

    d = layers.Conv2D(16, (1,1), padding='same', name=f"{name}_d_conv1x1")(x)
    d = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_d_act")(d)

    out = layers.Concatenate(axis=-1, name=f"{name}_concat")([a, b, c, d])
    return out


def build_se_lead_cnn(config=None):
    """
    Builds the SE-LEAD-CNN (LEAD-CNN + channel attention).

    Args:
        config: config module. If None, loads from config.py automatically.

    Returns:
        Uncompiled Keras Model
    """
    cfg = config or _cfg

    inputs = layers.Input(shape=IMG_SIZE, name="input")

    # ── Backbone: UNCHANGED from base LEAD-CNN ────────────────────────────────
    x = layers.Conv2D(32, (3,3), padding='valid', name="conv1")(inputs)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="act1")(x)
    x = layers.MaxPooling2D((2,2), name="pool1")(x)

    x = layers.Conv2D(32, (3,3), padding='valid', name="conv2")(x)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="act2")(x)
    x = layers.MaxPooling2D((2,2), name="pool2")(x)
    x = layers.Dropout(cfg.DROPOUT_CONV, name="drop2")(x)

    x = layers.Conv2D(64, (3,3), padding='valid', name="conv3")(x)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="act3")(x)
    x = layers.Conv2D(64, (3,3), padding='valid', name="conv4")(x)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="act4")(x)
    x = layers.MaxPooling2D((2,2), name="pool4")(x)
    x = layers.Dropout(cfg.DROPOUT_CONV, name="drop4")(x)

    x = layers.Conv2D(128, (3,3), padding='valid', name="conv5")(x)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="act5")(x)
    x = layers.Conv2D(128, (3,3), padding='valid', name="conv6")(x)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="act6")(x)
    x = layers.MaxPooling2D((2,2), name="pool6")(x)
    x = layers.Dropout(cfg.DROPOUT_CONV, name="drop6")(x)

    # ── Dimension Reduction Block: UNCHANGED ────────────────────────────────
    x = dimension_reduction_block(x, name="dim_red")   # 10x10x64

    # ── SE Block: inserted after concatenation, before flatten ────────────────
    x = squeeze_and_excitation(x, ratio=cfg.SE_REDUCTION_RATIO, name="se")

    # ── Classifier Head: UNCHANGED ────────────────────────────────────────────
    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(128, name="fc1")(x)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="fc1_act")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(cfg.DROPOUT_FC1, name="fc1_drop")(x)

    x = layers.Dense(64, name="fc2")(x)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="fc2_act")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Dropout(cfg.DROPOUT_FC2, name="fc2_drop")(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SE_LEAD_CNN")
    return model


if __name__ == "__main__":
    model = build_se_lead_cnn()
    model.summary()
    print(f"\nTotal parameters   : {model.count_params():,}")
    print(f"Base LEAD-CNN      : 1,132,612")
    print(f"Difference         : +{model.count_params() - 1132612:,} params")
    print(f"SE reduction ratio : {_cfg.SE_REDUCTION_RATIO}")
