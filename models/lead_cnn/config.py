# Configuration file for Lead CNN model

IMG_SIZE = (224, 224, 3)
NUM_CLASSES = 4


BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4


DATASET_PATH = "../../data/augmented_data"
MODEL_SAVE_PATH = "saved_weights/lead_cnn_model.keras"


RANDOM_SEED = 42