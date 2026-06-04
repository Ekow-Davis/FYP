"""
Depthwise Separable LEAD-CNN Architecture.

Changes vs original LEAD-CNN:
  - All 6 backbone Conv2D layers replaced with depthwise separable
    equivalents: DepthwiseConv2D (spatial filtering) + Conv2D 1x1 (channel mixing)
  - WIDTH_MULTIPLIER scales filter counts to control parameter budget
  - Modified Dimension Reduction Block is UNCHANGED — preserved exactly
    as in the original paper
  - Classifier head is UNCHANGED

Why depthwise separable convolutions:
  A standard Conv2D with F filters on C input channels costs:
      F × C × kH × kW parameters per layer
  A depthwise separable equivalent costs:
      C × kH × kW  (depthwise)  +  F × C × 1 × 1  (pointwise)
  For a 3×3 conv this is roughly 8-9x fewer parameters per layer.

Parameter comparison (WIDTH_MULTIPLIER=1.0):
  Standard LEAD-CNN backbone  : ~280,000 params
  DSC LEAD-CNN backbone       : ~35,000 params
  Dim reduction + head        : unchanged (~850,000 params)
  Total DSC LEAD-CNN          : ~885,000 params  (vs ~1,132,612 original)

At WIDTH_MULTIPLIER=0.75 the total drops further to ~700k.
"""

import sys
import os
import math

from tensorflow.keras import layers, models

_HERE   = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.join(_HERE, "..", "shared")
sys.path.insert(0, _SHARED)

from arch_config import IMG_SIZE, NUM_CLASSES
import importlib.util

# Load this experiment's config by file path to avoid any name collision
_cfg_spec = importlib.util.spec_from_file_location(
    "depthwise_config", os.path.join(_HERE, "config.py")
)
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)


def _scaled_filters(base_filters, multiplier):
    """Round filter count to nearest 8 — keeps depthwise ops efficient."""
    return max(8, int(math.ceil(base_filters * multiplier / 8) * 8))


def dsc_block(x, filters, name, padding='valid'):
    """
    Depthwise Separable Convolution block:
        DepthwiseConv2D (3x3) → pointwise Conv2D (1x1) → LeakyReLU
    Valid padding used to match original LEAD-CNN output shapes.
    """
    x = layers.DepthwiseConv2D(
        kernel_size=(3,3), padding=padding,
        use_bias=False, name=f"{name}_dw"
        )(x)
    x = layers.Conv2D(
        filters, (1,1), padding='same',
        use_bias=False, name=f"{name}_pw"
        )(x)
    x = layers.BatchNormalization(
        name=f"{name}_bn"
        )(x)    # ADDED THIS
    x = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_act")(x)
    return x


def dimension_reduction_block(x, name="dim_red"):
    """
    Modified Dimension Reduction Block — UNCHANGED from paper (Fig.4).
    4 parallel branches concatenated to 10x10x64.
    Branch structure verified against Table 2 parameter counts.
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


def build_dsc_lead_cnn(config=None):
    """
    Builds the Depthwise Separable LEAD-CNN.

    Args:
        config: config module. If None, loads from config.py automatically.

    Returns:
        Uncompiled Keras Model
    """
    cfg = config or _cfg
    m   = cfg.WIDTH_MULTIPLIER

    inputs = layers.Input(shape=IMG_SIZE, name="input")

    # ── Block 1 ── filters: 32 → scaled ──────────────────────────────────────
    f1 = _scaled_filters(32, m)
    x  = dsc_block(inputs, f1, name="dsc1")           # 222x222xf1
    x  = layers.MaxPooling2D((2,2), name="pool1")(x)   # 111x111xf1

    # ── Block 2 ──────────────────────────────────────────────────────────────
    x  = dsc_block(x, f1, name="dsc2")                # 109x109xf1
    x  = layers.MaxPooling2D((2,2), name="pool2")(x)   # 54x54xf1
    x  = layers.Dropout(cfg.DROPOUT_CONV, name="drop2")(x)

    # ── Block 3 ── filters: 64 → scaled ──────────────────────────────────────
    f2 = _scaled_filters(64, m)
    x  = dsc_block(x, f2, name="dsc3")                # 52x52xf2
    x  = dsc_block(x, f2, name="dsc4")                # 50x50xf2
    x  = layers.MaxPooling2D((2,2), name="pool4")(x)   # 25x25xf2
    x  = layers.Dropout(cfg.DROPOUT_CONV, name="drop4")(x)

    # ── Block 4 ── filters: 128 → scaled ─────────────────────────────────────
    f3 = _scaled_filters(128, m)
    x  = dsc_block(x, f3, name="dsc5")                # 23x23xf3
    x  = dsc_block(x, f3, name="dsc6")                # 21x21xf3
    x  = layers.MaxPooling2D((2,2), name="pool6")(x)   # 10x10xf3
    x  = layers.Dropout(cfg.DROPOUT_CONV, name="drop6")(x)

    # ── Dimension Reduction Block (UNCHANGED from paper) ──────────────────────
    # Note: if WIDTH_MULTIPLIER != 1.0, the input channel count to this block
    # will differ from the original 128. The block's 1x1 convs handle any
    # input channel count so it adapts automatically.
    x  = dimension_reduction_block(x, name="dim_red")  # 10x10x64

    # ── Classifier Head (UNCHANGED from paper) ────────────────────────────────
    x  = layers.Flatten(name="flatten")(x)              # 6400

    x  = layers.Dense(128, name="fc1")(x)
    x  = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="fc1_act")(x)
    x  = layers.BatchNormalization(name="fc1_bn")(x)
    x  = layers.Dropout(cfg.DROPOUT_FC1, name="fc1_drop")(x)

    x  = layers.Dense(64, name="fc2")(x)
    x  = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="fc2_act")(x)
    x  = layers.BatchNormalization(name="fc2_bn")(x)
    x  = layers.Dropout(cfg.DROPOUT_FC2, name="fc2_drop")(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs,
                         name=f"DSC_LEAD_CNN_w{m}")
    return model


if __name__ == "__main__":
    model = build_dsc_lead_cnn()
    model.summary()
    print(f"\nTotal parameters   : {model.count_params():,}")
    print(f"Base LEAD-CNN      : 1,132,612")
    print(f"Reduction          : {1132612 - model.count_params():,} fewer params")
    print(f"Width multiplier   : {_cfg.WIDTH_MULTIPLIER}")
