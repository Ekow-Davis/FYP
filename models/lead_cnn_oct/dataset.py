import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMG_SIZE, BATCH_SIZE, DATASET_PATH, RANDOM_SEED

AUGMENTATION = "oct_conservative"  # options: "paper", "none", "modern", "oct_conservative"


def _build_dataframe(split):
    rows = []
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
        # OCT images have fixed anatomical orientation (retina top, choroid bottom)
        # vertical flip and 90° rotation are anatomically invalid for this domain
        return ImageDataGenerator(
            rescale=1.0 / 255,
            horizontal_flip=True,
        )
    elif AUGMENTATION == "oct_conservative":
        # Domain-appropriate augmentation for OCT retinal images
        return ImageDataGenerator(
            rescale=1.0 / 255,
            horizontal_flip=True,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='reflect',
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

    train_gen = _make_datagen().flow_from_dataframe(
        _build_dataframe("train"),
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=RANDOM_SEED,
    )

    val_gen = eval_datagen.flow_from_dataframe(
        _build_dataframe("val"),
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=RANDOM_SEED,
    )

    test_gen = eval_datagen.flow_from_dataframe(
        _build_dataframe("test"),
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=RANDOM_SEED,
    )

    return train_gen, val_gen, test_gen
