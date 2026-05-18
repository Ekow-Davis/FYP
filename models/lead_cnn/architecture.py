"""
LEAD-CNN Architecture — exact replication of Table 2 (paper)

Layer-by-layer structure (Table 2):

  Input:         224x224x3
  Conv1 (3x3,32) -> LeakyReLU -> MaxPool 2x2       => 111x111x32
  Conv2 (3x3,32) -> LeakyReLU -> MaxPool 2x2
                -> Dropout(0.25)                    =>  54x54x32
  Conv3 (3x3,64) -> LeakyReLU                      =>  52x52x64
  Conv4 (3x3,64) -> LeakyReLU -> MaxPool 2x2
                -> Dropout(0.25)                    =>  25x25x64
  Conv5 (3x3,128)-> LeakyReLU                      =>  23x23x128
  Conv6 (3x3,128)-> LeakyReLU -> MaxPool 2x2
                -> Dropout(0.25)                    =>  10x10x128

  [Modified Dimension Reduction Block]             =>  10x10x64
    (see dimension_reduction_block.py)

  Flatten                                           =>  6400
  Dense(128) -> LeakyReLU -> BatchNorm
             -> Dropout(0.25)
  Dense(64)  -> LeakyReLU -> BatchNorm
             -> Dropout(0.5)
  Dense(4)   -> Softmax

Notes:
  - All LeakyReLU use alpha=0.2 (from Fig. 4; Eq.1 alpha=0.01 is a typo in the paper)
  - All Conv layers use valid padding (no padding) to match output shapes in Table 2
  - MaxPool always 2x2 in the backbone
  - No padding on backbone convs: 224->222->111, 111->109->54, 54->52, 52->50->25, etc.
"""

from tensorflow.keras import layers, models
from dimension_reduction_block import dimension_reduction_block
from config import IMG_SIZE, NUM_CLASSES


LEAKY_ALPHA = 0.2  # Fig. 4 specifies alpha=0.2


def build_lead_cnn():
    inputs = layers.Input(shape=IMG_SIZE, name="input")

    # ── Block 1 ───────
    # Conv1: 3x3, 32 filters, valid => 222x222x32
    x = layers.Conv2D(32, (3, 3), padding='valid', name="conv1")(inputs)
    x = layers.LeakyReLU(negative_slope=LEAKY_ALPHA, name="act1")(x)
    # MaxPool 2x2 => 111x111x32
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # ── Block 2 ──────────────
    # Conv2: 3x3, 32 filters, valid => 109x109x32
    x = layers.Conv2D(32, (3, 3), padding='valid', name="conv2")(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_ALPHA, name="act2")(x)
    # MaxPool 2x2 => 54x54x32
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # ── Block 3 ──────────────
    # Conv3: 3x3, 64 filters, valid => 52x52x64
    x = layers.Conv2D(64, (3, 3), padding='valid', name="conv3")(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_ALPHA, name="act3")(x)

    # Conv4: 3x3, 64 filters, valid => 50x50x64
    x = layers.Conv2D(64, (3, 3), padding='valid', name="conv4")(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_ALPHA, name="act4")(x)
    # MaxPool 2x2 => 25x25x64
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.Dropout(0.25, name="drop4")(x)

    # ── Block 4 ─────────
    # Conv5: 3x3, 128 filters, valid => 23x23x128
    x = layers.Conv2D(128, (3, 3), padding='valid', name="conv5")(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_ALPHA, name="act5")(x)

    # Conv6: 3x3, 128 filters, valid => 21x21x128
    x = layers.Conv2D(128, (3, 3), padding='valid', name="conv6")(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_ALPHA, name="act6")(x)
    # MaxPool 2x2 => 10x10x128
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool6")(x)
    x = layers.Dropout(0.25, name="drop6")(x)

    # ── Modified Dimension Reduction Block ──────────
    # Input: 10x10x128 -> Output: 10x10x64
    x = dimension_reduction_block(x, name="dim_red")

    # ── Classifier Head ──────
    x = layers.Flatten(name="flatten")(x)                          # 6400

    x = layers.Dense(128, name="fc1")(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_ALPHA, name="fc1_act")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(0.25, name="fc1_drop")(x)

    x = layers.Dense(64, name="fc2")(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_ALPHA, name="fc2_act")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Dropout(0.5, name="fc2_drop")(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="LEAD_CNN")
    return model


if __name__ == "__main__":
    model = build_lead_cnn()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    print("Expected from paper: ~1,132,612")
