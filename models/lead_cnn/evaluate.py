import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from dataset import create_generators
from config import MODEL_SAVE_PATH



def main():
  _, _, test_gen = create_generators()


  model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)


  predictions = model.predict(test_gen)
  y_pred = np.argmax(predictions, axis=1)
  y_true = test_gen.classes


  print("\nClassification Report:")
  print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))


  print("\nConfusion Matrix:")
  print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
  main()