"""
Modified Dimension Reduction Block — LEAD-CNN (Fig. 4, Table 2)

4 parallel branches, all receiving the previous layer output (10x10x128).
Branch structure verified against Table 2 parameter counts:

  Branch A: Conv7  1x1 128->16 (2064 params) -> LeakyReLU
            Conv10 3x3 16->16  (2320 params) -> LeakyReLU

  Branch B: Conv8  1x1 128->16 (2064 params) -> LeakyReLU
            Conv11 5x5 16->16  (6416 params) -> LeakyReLU

  Branch C: MaxPool 3x3 (stride=1, same padding)
            Conv9  1x1 128->16 (2064 params) -> LeakyReLU

  Branch D: Conv12 1x1 128->16 (2064 params) -> LeakyReLU

  Concatenate all 4 branches -> 10x10x64

Parameter check:
  2064 = 16*(1*1*128 + 1)  ✓
  2320 = 16*(3*3*16  + 1)  ✓
  6416 = 16*(5*5*16  + 1)  ✓
"""

from tensorflow.keras import layers


def dimension_reduction_block(x, name="dim_red"):
    """
    Modified Dimension Reduction Block as described in LEAD-CNN paper.

    Args:
        x:    Input tensor, expected shape (batch, 10, 10, 128)
        name: Prefix for layer names (keeps graph readable)

    Returns:
        Concatenated output tensor, shape (batch, 10, 10, 64)
    """

    # --- Branch A: 1x1 -> 3x3 ---
    a = layers.Conv2D(16, (1, 1), padding='same', name=f"{name}_a_conv1x1")(x)
    a = layers.LeakyReLU(negative_slope=0.2, name=f"{name}_a_act1")(a)
    a = layers.Conv2D(16, (3, 3), padding='same', name=f"{name}_a_conv3x3")(a)
    a = layers.LeakyReLU(negative_slope=0.2, name=f"{name}_a_act2")(a)

    # --- Branch B: 1x1 -> 5x5 ---
    b = layers.Conv2D(16, (1, 1), padding='same', name=f"{name}_b_conv1x1")(x)
    b = layers.LeakyReLU(negative_slope=0.2, name=f"{name}_b_act1")(b)
    b = layers.Conv2D(16, (5, 5), padding='same', name=f"{name}_b_conv5x5")(b)
    b = layers.LeakyReLU(negative_slope=0.2, name=f"{name}_b_act2")(b)

    # --- Branch C: MaxPool 3x3 -> 1x1 ---
    c = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same',
                             name=f"{name}_c_pool")(x)
    c = layers.Conv2D(16, (1, 1), padding='same', name=f"{name}_c_conv1x1")(c)
    c = layers.LeakyReLU(negative_slope=0.2, name=f"{name}_c_act")(c)

    # --- Branch D: standalone 1x1 ---
    d = layers.Conv2D(16, (1, 1), padding='same', name=f"{name}_d_conv1x1")(x)
    d = layers.LeakyReLU(negative_slope=0.2, name=f"{name}_d_act")(d)

    # --- Concatenate: 4 x 16 filters = 64 ---
    out = layers.Concatenate(axis=-1, name=f"{name}_concat")([a, b, c, d])
    return out
