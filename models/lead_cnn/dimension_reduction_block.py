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

def dimension_reduction_block(x, name="dim_red"):
    # Path 1: 1x1 Conv
    path1 = layers.Conv2D(16, (1, 1), padding='same', name=f"{name}_p1_conv")(x)
    path1 = layers.LeakyReLU(alpha=0.2, name=f"{name}_p1_act")(path1)
    
    # Path 2: 1x1 Conv -> 3x3 Conv
    path2 = layers.Conv2D(16, (1, 1), padding='same')(x)
    path2 = layers.LeakyReLU(alpha=0.2)(path2)
    path2 = layers.Conv2D(16, (3, 3), padding='same')(path2)
    path2 = layers.LeakyReLU(alpha=0.2)(path2)
    
    # Path 3: 1x1 Conv -> 5x5 Conv
    path3 = layers.Conv2D(16, (1, 1), padding='same')(x)
    path3 = layers.LeakyReLU(alpha=0.2)(path3)
    path3 = layers.Conv2D(16, (5, 5), padding='same')(path3)
    path3 = layers.LeakyReLU(alpha=0.2)(path3)
    
    # Path 4: 3x3 MaxPool -> 1x1 Conv
    path4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='same')(x)
    path4 = layers.Conv2D(16, (1, 1), padding='same')(path4)
    path4 = layers.LeakyReLU(alpha=0.2)(path4)
    
    # Concatenate all 4 paths
    concat = layers.Concatenate(axis=-1)([path1, path2, path3, path4])
    return concat