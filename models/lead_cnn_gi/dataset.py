import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMG_SIZE, BATCH_SIZE, DATASET_PATH, RANDOM_SEED

AUGMENTATION = "paper"  # options: "paper", "none", "modern"


def _build_dataframe(splits):
    """Build a DataFrame of (filepath, label) rows from one or more split dirs."""
    rows = []
    for split in splits:
        split_dir = os.path.join(DATASET_PATH, split)
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    rows.append({
                        "filepath": os.path.join(class_dir, fname),
                        "label": class_name,
                    })
    return pd.DataFrame(rows)


def _make_datagen():
    if AUGMENTATION == "paper":
        return ImageDataGenerator(
            rescale=1.0 / 255,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=90,
        )
    elif AUGMENTATION == "none":
        return ImageDataGenerator(rescale=1.0 / 255)
    elif AUGMENTATION == "modern":
        return ImageDataGenerator(
            rescale=1.0 / 255,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.15,
            shear_range=0.1,
            brightness_range=(0.8, 1.2),
            fill_mode="reflect",
        )
    else:
        raise ValueError(f"Unknown AUGMENTATION mode: {AUGMENTATION!r}")


def create_generators():
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Paper Table 11: train+val combined = 1300/class for training
    train_df = _build_dataframe(["train", "val"])
    train_gen = _make_datagen().flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=RANDOM_SEED,
    )

    # test/ used as validation monitor during training (200/class)
    test_df = _build_dataframe(["test"])
    val_gen = eval_datagen.flow_from_dataframe(
        test_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=RANDOM_SEED,
    )

    test_gen = eval_datagen.flow_from_dataframe(
        test_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=RANDOM_SEED,
    )

    return train_gen, val_gen, test_gen
