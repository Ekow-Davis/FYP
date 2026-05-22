import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from architecture import build_lead_cnn
from dataset import create_generators
from config import LEARNING_RATE, EPOCHS, MODEL_SAVE_PATH, RESULTS_PATH


def main(experiment_name="experiment"):
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    train_gen, val_gen, _ = create_generators()
    model = build_lead_cnn()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    csv_path = os.path.join(RESULTS_PATH, f"{experiment_name}_history.csv")
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        CSVLogger(csv_path, append=False),
    ]

    start = time.time()
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
    elapsed = time.time() - start

    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\nTotal training time: {minutes}m {seconds}s")
    print(f"Epoch history saved to: {csv_path}")


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "experiment"
    main(experiment_name=name)
