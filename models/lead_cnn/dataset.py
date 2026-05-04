import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMG_SIZE, BATCH_SIZE, DATASET_PATH


def _augment(image):
    # Gaussian noise
    if np.random.random() < 0.5:
        noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    # Brightness shift
    if np.random.random() < 0.5:
        factor = np.random.uniform(0.85, 1.15)
        image = np.clip(image * factor, 0.0, 1.0)

    # Contrast shift
    if np.random.random() < 0.5:
        mean = np.mean(image)
        factor = np.random.uniform(0.85, 1.15)
        image = np.clip((image - mean) * factor + mean, 0.0, 1.0)

    # Cutout: mask one random patch
    if np.random.random() < 0.5:
        h, w = image.shape[:2]
        ph = np.random.randint(20, 40)
        pw = np.random.randint(20, 40)
        y = np.random.randint(0, h - ph)
        x = np.random.randint(0, w - pw)
        image[y:y + ph, x:x + pw] = 0.0

    return image


def create_generators():
    train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=_augment)
    eval_datagen  = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        f"{DATASET_PATH}/train",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_gen = eval_datagen.flow_from_directory(
        f"{DATASET_PATH}/val",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_gen = eval_datagen.flow_from_directory(
        f"{DATASET_PATH}/test",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen
