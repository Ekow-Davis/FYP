"""
Combined LEAD-CNN: Depthwise Separable Convolutions + SE Channel Attention.

Changes vs original LEAD-CNN:
  - Backbone: DSC (same as depthwise/ experiment)
  - Dimension Reduction Block: UNCHANGED
  - SE block inserted after dim reduction block (same as attention/ experiment)
  - Classifier head: UNCHANGED
"""

import sys
import os
import math
import importlib.util

from tensorflow.keras import layers, models

_HERE   = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.join(_HERE, "..", "shared")
sys.path.insert(0, _SHARED)

from arch_config import IMG_SIZE, NUM_CLASSES

_cfg_spec = importlib.util.spec_from_file_location(
    "combined_config", os.path.join(_HERE, "config.py")
)
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)


def _scaled_filters(base_filters, multiplier):
    return max(8, int(math.ceil(base_filters * multiplier / 8) * 8))


def dsc_block(x, filters, name, padding='valid'):
    x = layers.DepthwiseConv2D(
        kernel_size=(3,3), padding=padding,
        use_bias=False, name=f"{name}_dw"
    )(x)
    x = layers.Conv2D(
        filters, (1,1), padding='same',
        use_bias=True, name=f"{name}_pw"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_act")(x)
    return x


def squeeze_and_excitation(x, ratio, name="se"):
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_squeeze")(x)
    se = layers.Dense(max(1, channels // ratio),
                      activation='relu', name=f"{name}_fc1")(se)
    se = layers.Dense(channels,
                      activation='sigmoid', name=f"{name}_fc2")(se)
    se = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


def dimension_reduction_block(x, name="dim_red"):
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

    return layers.Concatenate(axis=-1, name=f"{name}_concat")([a, b, c, d])


def build_combined_lead_cnn(config=None):
    cfg = config or _cfg
    m   = cfg.WIDTH_MULTIPLIER

    inputs = layers.Input(shape=IMG_SIZE, name="input")

    # ── DSC Backbone ──────────────────────────────────────────────────────────
    f1 = _scaled_filters(32, m)
    x  = dsc_block(inputs, f1, name="dsc1")
    x  = layers.MaxPooling2D((2,2), name="pool1")(x)

    x  = dsc_block(x, f1, name="dsc2")
    x  = layers.MaxPooling2D((2,2), name="pool2")(x)
    x  = layers.Dropout(cfg.DROPOUT_CONV, name="drop2")(x)

    f2 = _scaled_filters(64, m)
    x  = dsc_block(x, f2, name="dsc3")
    x  = dsc_block(x, f2, name="dsc4")
    x  = layers.MaxPooling2D((2,2), name="pool4")(x)
    x  = layers.Dropout(cfg.DROPOUT_CONV, name="drop4")(x)

    f3 = _scaled_filters(128, m)
    x  = dsc_block(x, f3, name="dsc5")
    x  = dsc_block(x, f3, name="dsc6")
    x  = layers.MaxPooling2D((2,2), name="pool6")(x)
    x  = layers.Dropout(cfg.DROPOUT_CONV, name="drop6")(x)

    # ── Dimension Reduction Block: UNCHANGED ──────────────────────────────────
    x = dimension_reduction_block(x, name="dim_red")   # 10x10x64

    # ── SE Channel Attention ──────────────────────────────────────────────────
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

    model = models.Model(inputs=inputs, outputs=outputs,
                         name=f"Combined_LEAD_CNN_w{m}_se{cfg.SE_REDUCTION_RATIO}")
    return model


if __name__ == "__main__":
    model = build_combined_lead_cnn()
    model.summary()
    print(f"\nTotal parameters   : {model.count_params():,}")
    print(f"Base LEAD-CNN      : 1,132,612")
    diff = model.count_params() - 1132612
    sign = "+" if diff >= 0 else ""
    print(f"Difference         : {sign}{diff:,}")
