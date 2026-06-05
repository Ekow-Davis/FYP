"""
DSC-DimRed LEAD-CNN Architecture.

Changes vs original LEAD-CNN:
  - Backbone: UNCHANGED (6 standard Conv2D layers, identical to paper)
  - Modified Dimension Reduction Block:
      Branch A: 1x1 (standard) → DSC 3x3 → LeakyReLU [was standard 3x3]
      Branch B: 1x1 (standard) → DSC 5x5 → LeakyReLU [was standard 5x5]
      Branch C: MaxPool → 1x1 (standard) → LeakyReLU  [UNCHANGED]
      Branch D: 1x1 (standard) → LeakyReLU             [UNCHANGED]
  - Classifier head: UNCHANGED

Why only 3x3 and 5x5 in the block:
  The 1x1 convolutions are already the most efficient possible operation
  for channel mixing — depthwise can't replace them since depthwise doesn't
  mix channels. The MaxPool branch is also already lightweight. The 3x3 and
  5x5 convolutions are the only operations in the block that benefit from
  the DSC factorisation.

Parameter impact:
  Original block 3x3 branch second conv: 16*(3*3*16+1) = 2,320 params
  DSC equivalent: 16*(3*3)+16*(1*16+1) = 144+272 = 416 params  (-82%)

  Original block 5x5 branch second conv: 16*(5*5*16+1) = 6,416 params
  DSC equivalent: 16*(5*5)+16*(1*16+1) = 400+272 = 672 params  (-90%)

  Saving: ~7,648 params in the block alone.
  Total model: slightly fewer than original ~1,132,612.

BatchNorm added after DSC layers when DSC_USE_BN=True (recommended).
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
    "dsc_dimred_config", os.path.join(_HERE, "config.py")
)
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)


def _dsc_conv(x, filters, kernel_size, name, use_bn=True):
    """
    Depthwise separable conv for use inside the dim reduction block.
    DepthwiseConv2D → pointwise Conv2D 1x1 → (optional BN) → LeakyReLU

    Args:
        x:           input tensor
        filters:     output channel count (pointwise)
        kernel_size: spatial kernel for depthwise step (3 or 5)
        name:        layer name prefix
        use_bn:      whether to add BatchNorm after pointwise conv
    """
    x = layers.DepthwiseConv2D(
        kernel_size=(kernel_size, kernel_size),
        padding='same',
        use_bias=False,
        name=f"{name}_dw"
    )(x)
    x = layers.Conv2D(
        filters, (1, 1),
        padding='same',
        use_bias=False,
        name=f"{name}_pw"
    )(x)
    if use_bn:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_act")(x)
    return x


def dimension_reduction_block_dsc(x, name="dim_red"):
    """
    Modified Dimension Reduction Block with DSC in 3x3 and 5x5 branches.

    Branch A: Conv 1x1 → LeakyReLU → DSC 3x3 → (BN) → LeakyReLU
    Branch B: Conv 1x1 → LeakyReLU → DSC 5x5 → (BN) → LeakyReLU
    Branch C: MaxPool 3x3 → Conv 1x1 → LeakyReLU          [UNCHANGED]
    Branch D: Conv 1x1 → LeakyReLU                         [UNCHANGED]

    All branches output 10x10x16, concatenated → 10x10x64
    """
    use_bn = _cfg.DSC_USE_BN

    # ── Branch A: 1x1 bottleneck → DSC 3x3 ───────────────────────────────────
    a = layers.Conv2D(16, (1, 1), padding='same', name=f"{name}_a_conv1x1")(x)
    a = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_a_act1")(a)
    a = _dsc_conv(a, 16, kernel_size=3, name=f"{name}_a_dsc3x3", use_bn=use_bn)

    # ── Branch B: 1x1 bottleneck → DSC 5x5 ───────────────────────────────────
    b = layers.Conv2D(16, (1, 1), padding='same', name=f"{name}_b_conv1x1")(x)
    b = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_b_act1")(b)
    b = _dsc_conv(b, 16, kernel_size=5, name=f"{name}_b_dsc5x5", use_bn=use_bn)

    # ── Branch C: MaxPool → 1x1 (UNCHANGED) ──────────────────────────────────
    c = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same',
                            name=f"{name}_c_pool")(x)
    c = layers.Conv2D(16, (1, 1), padding='same', name=f"{name}_c_conv1x1")(c)
    c = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_c_act")(c)

    # ── Branch D: standalone 1x1 (UNCHANGED) ─────────────────────────────────
    d = layers.Conv2D(16, (1, 1), padding='same', name=f"{name}_d_conv1x1")(x)
    d = layers.LeakyReLU(negative_slope=_cfg.LEAKY_ALPHA, name=f"{name}_d_act")(d)

    # ── Concatenate: 4 x 16 = 64 channels ────────────────────────────────────
    return layers.Concatenate(axis=-1, name=f"{name}_concat")([a, b, c, d])


def build_dsc_dimred_lead_cnn(config=None):
    """
    Builds the DSC-DimRed LEAD-CNN.
    Standard backbone + DSC in dim reduction block only.
    """
    cfg = config or _cfg

    inputs = layers.Input(shape=IMG_SIZE, name="input")

    # ── Backbone: UNCHANGED from paper ───────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding='valid', name="conv1")(inputs)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="act1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(32, (3, 3), padding='valid', name="conv2")(x)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="act2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(cfg.DROPOUT_CONV, name="drop2")(x)

    x = layers.Conv2D(64, (3, 3), padding='valid', name="conv3")(x)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="act3")(x)
    x = layers.Conv2D(64, (3, 3), padding='valid', name="conv4")(x)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="act4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = layers.Dropout(cfg.DROPOUT_CONV, name="drop4")(x)

    x = layers.Conv2D(128, (3, 3), padding='valid', name="conv5")(x)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="act5")(x)
    x = layers.Conv2D(128, (3, 3), padding='valid', name="conv6")(x)
    x = layers.LeakyReLU(negative_slope=cfg.LEAKY_ALPHA, name="act6")(x)
    x = layers.MaxPooling2D((2, 2), name="pool6")(x)
    x = layers.Dropout(cfg.DROPOUT_CONV, name="drop6")(x)

    # ── Modified Dim Reduction Block with DSC in 3x3 and 5x5 branches ────────
    x = dimension_reduction_block_dsc(x, name="dim_red")   # 10x10x64

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
                         name="DSC_DimRed_LEAD_CNN")
    return model


if __name__ == "__main__":
    model = build_dsc_dimred_lead_cnn()
    model.summary()
    total = model.count_params()
    print(f"\nTotal parameters      : {total:,}")
    print(f"Base LEAD-CNN         : 1,132,612")
    print(f"Difference            : {total - 1132612:+,}")
    print(f"DSC_USE_BN            : {_cfg.DSC_USE_BN}")
