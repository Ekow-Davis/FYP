import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMG_SIZE, BATCH_SIZE, DATASET_PATH




def create_generators():
  datagen = ImageDataGenerator(rescale=1./255)


  train_gen = datagen.flow_from_directory(
    f"{DATASET_PATH}/train",
    target_size=IMG_SIZE[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical'
  )


  val_gen = datagen.flow_from_directory(
    f"{DATASET_PATH}/val",
    target_size=IMG_SIZE[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical'
  )


  test_gen = datagen.flow_from_directory(
    f"{DATASET_PATH}/test",
    target_size=IMG_SIZE[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
  )


  return train_gen, val_gen, test_gen