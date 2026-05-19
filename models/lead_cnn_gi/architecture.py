from tensorflow.keras import layers, models
from dimension_reduction_block import dimension_reduction_block
from config import IMG_SIZE, NUM_CLASSES


def build_lead_cnn():
    inputs = layers.Input(shape=IMG_SIZE)

    x = layers.Conv2D(32, (3, 3), padding='valid', name="conv_1")(inputs)
    x = layers.LeakyReLU(alpha=0.2, name="leakyrelu_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_1")(x)

    x = layers.Conv2D(32, (3, 3), padding='valid', name="conv_2")(x)
    x = layers.LeakyReLU(alpha=0.2, name="leakyrelu_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_2")(x)
    x = layers.Dropout(0.25, name="dropout_1")(x)

    x = layers.Conv2D(64, (3, 3), padding='valid', name="conv_3")(x)
    x = layers.LeakyReLU(alpha=0.2, name="leakyrelu_3")(x)

    x = layers.Conv2D(64, (3, 3), padding='valid', name="conv_4")(x)
    x = layers.LeakyReLU(alpha=0.2, name="leakyrelu_4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_3")(x)
    x = layers.Dropout(0.25, name="dropout_2")(x)

    x = layers.Conv2D(128, (3, 3), padding='valid', name="conv_5")(x)
    x = layers.LeakyReLU(alpha=0.2, name="leakyrelu_5")(x)

    x = layers.Conv2D(128, (3, 3), padding='valid', name="conv_6")(x)
    x = layers.LeakyReLU(alpha=0.2, name="leakyrelu_6")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_4")(x)
    x = layers.Dropout(0.25, name="dropout_3")(x)

    x = dimension_reduction_block(x, filters=16)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, name="fc1")(x)
    x = layers.LeakyReLU(alpha=0.2, name="fc1_lrelu")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(0.25, name="fc1_dropout")(x)

    x = layers.Dense(64, name="fc2")(x)
    x = layers.LeakyReLU(alpha=0.2, name="fc2_lrelu")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Dropout(0.5, name="fc2_dropout")(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="output")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="LEAD_CNN_GI")
