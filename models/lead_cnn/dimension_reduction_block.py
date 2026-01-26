# models/lead_cnn/dimension_reduction_block.py

import tensorflow as tf
from tensorflow.keras import layers


def dimension_reduction_block(x, filters=64, name="dimension_reduction"):
  """
    Dimension Reduction Block (as described in LEAD-CNN paper)
    - 1x1 convolution for channel compression
    - LeakyReLU activation
    - Batch normalization
    - Max pooling
  """


  x = layers.Conv2D(filters, (1, 1), padding='same', name=f"{name}_conv1x1")(x)
  x = layers.LeakyReLU(alpha=0.1, name=f"{name}_leakyrelu")(x)
  x = layers.BatchNormalization(name=f"{name}_bn")(x)
  x = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool")(x)


  return x