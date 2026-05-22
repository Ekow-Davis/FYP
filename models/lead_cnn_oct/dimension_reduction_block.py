from tensorflow.keras import layers


def dimension_reduction_block(x, filters=16, name="dim_reduction"):
    # Branch A: 1x1 -> 3x3
    a = layers.Conv2D(filters, (1, 1), padding='same', name=f"{name}_a_1x1")(x)
    a = layers.LeakyReLU(alpha=0.2, name=f"{name}_a_lrelu1")(a)
    a = layers.Conv2D(filters, (3, 3), padding='same', name=f"{name}_a_3x3")(a)
    a = layers.LeakyReLU(alpha=0.2, name=f"{name}_a_lrelu2")(a)

    # Branch B: 1x1 -> 5x5
    b = layers.Conv2D(filters, (1, 1), padding='same', name=f"{name}_b_1x1")(x)
    b = layers.LeakyReLU(alpha=0.2, name=f"{name}_b_lrelu1")(b)
    b = layers.Conv2D(filters, (5, 5), padding='same', name=f"{name}_b_5x5")(b)
    b = layers.LeakyReLU(alpha=0.2, name=f"{name}_b_lrelu2")(b)

    # Branch C: MaxPool 3x3 -> 1x1
    c = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=f"{name}_c_pool")(x)
    c = layers.Conv2D(filters, (1, 1), padding='same', name=f"{name}_c_1x1")(c)
    c = layers.LeakyReLU(alpha=0.2, name=f"{name}_c_lrelu")(c)

    # Branch D: 1x1 only
    d = layers.Conv2D(filters, (1, 1), padding='same', name=f"{name}_d_1x1")(x)
    d = layers.LeakyReLU(alpha=0.2, name=f"{name}_d_lrelu")(d)

    out = layers.Concatenate(name=f"{name}_concat")([a, b, c, d])
    return out
