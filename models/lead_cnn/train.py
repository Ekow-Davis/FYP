import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from architecture import build_lead_cnn
from dataset import create_generators
from config import LEARNING_RATE, EPOCHS, MODEL_SAVE_PATH




def main():
  train_gen, val_gen, _ = create_generators()

  model = build_lead_cnn()

  model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  model.summary()

  history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
  )

  model.save(MODEL_SAVE_PATH)
  print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
  main()