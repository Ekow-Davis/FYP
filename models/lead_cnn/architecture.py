import tensorflow as tf
from tensorflow.keras import layers, models
from dimension_reduction_block import dimension_reduction_block
from config import IMG_SIZE, NUM_CLASSES

# The paper does not follow a single repeating pattern so this can be removed
# def conv_block(x, filters, block_id):
#   x = layers.Conv2D(filters, (3, 3), padding='same', name=f"conv_{block_id}")(x)
#   x = layers.LeakyReLU(alpha=0.1, name=f"leakyrelu_{block_id}")(x)
#   x = layers.MaxPooling2D(pool_size=(2, 2), name=f"pool_{block_id}")(x)
#   x = layers.Dropout(0.25, name=f"dropout_{block_id}")(x)
#   return x


def build_lead_cnn():
  inputs = layers.Input(shape=IMG_SIZE)


  # Feature extraction blocks
  # x = conv_block(inputs, 32, 1)
  # x = conv_block(x, 64, 2)
  # x = conv_block(x, 128, 3)
  # x = conv_block(x, 256, 4)


  #New feature extraction block
  #Conv1: 3x3, 32 filters - LReLU - MaxPool (no dropout)
  x = layers.Conv2D(32, (3,3,), padding='valid', name="conv_1")(inputs)
  x = layers.LeakyReLU(alpha=0.2, name="leakyrelu_1")(x)
  x = layers.MaxPooling2D(pool_size=(2,2), name="pool_1")(x)

  #Conv2: 3x3, 32 filters - LReLU - MaxPool - Dropout
  x = layers.Conv2D(32, (3,3), padding="valid", name="conv_2")(x)
  x = layers.LeakyReLU(alpha=0.2, name="leakyrelu_2")(x)
  x = layers.MaxPooling2D(pool_size=(2,2), name="pool_2")(x)
  x = layers.Dropout(0.25, name="dropout_1")(x)

  #Conv3: 3x3, 64 filters - LReLU (no pool)
  x = layers.Conv2D(64, (3,3), padding="valid", name="conv_3")(x)
  x = layers.LeakyReLU(alpha=0.2, name="leakyrelu_3")(x)

  #Conv4: 3x3, 64 filters - LReLU - MaxPool - Dropout
  x = layers.Conv2D(64, (3,3), padding="valid", name="conv_4")(x)
  x = layers.LeakyReLU(alpha=0.2, name="leakyrelu_4")(x)
  x = layers.MaxPooling2D(pool_size=(2,2), name="pool_3")(x)
  x = layers.Dropout(0.25, name="dropout_2")(x)

  #Conv5: 3x3, 128 filters - LReLU (no pool)
  x = layers.Conv2D(128, (3,3), padding="valid", name="conv_5")(x)
  x = layers.LeakyReLU(alpha=0.2, name="leakyrelu_5")(x)

  #Conv6: 3x3, 128 filters - LReLU - MaxPool - Dropout
  x = layers.Conv2D(128, (3,3), padding="valid", name="conv_6")(x)
  x = layers.LeakyReLU(alpha=0.2, name="leakyrelu_6")(x)
  x = layers.MaxPooling2D(pool_size=(2,2), name="pool_4")(x)
  x = layers.Dropout(0.25, name="dropout_3")(x)



  # Dimension reduction block (paper's contribution)
  x = dimension_reduction_block(x, filters=16)


  # Classification head
  x = layers.Flatten(name="flatten")(x)
  x = layers.Dense(256, name="fc1")(x)
  x = layers.LeakyReLU(alpha=0.1, name="fc1_leakyrelu")(x)
  x = layers.Dropout(0.5, name="fc1_dropout")(x)


  outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="output")(x)


  model = models.Model(inputs=inputs, outputs=outputs, name="LEAD_CNN")


  return model




if __name__ == "__main__":
  model = build_lead_cnn()
  model.summary()