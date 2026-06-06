# Configuration file for Lead CNN model

IMG_SIZE = (224, 224, 3)
NUM_CLASSES = 4


BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3


DATASET_PATH = "../../data/cleaned_data_aug_gamma"
MODEL_SAVE_PATH = "saved_weights/lead_cnn_model.keras"


RANDOM_SEED = 42