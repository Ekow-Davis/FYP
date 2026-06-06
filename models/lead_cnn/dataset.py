import numpy as np
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from config import IMG_SIZE, BATCH_SIZE, DATASET_PATH, RANDOM_SEED

# Controls on-the-fly augmentation during training.
# Set to [] when training on a pre-generated augmented dataset
# (i.e. after running augment_dataset.py and updating DATASET_PATH in config.py).
#
# Available modes: "paper", "elastic", "clahe", "gamma", "blur", "cutout"
# Only one should be active at a time to match single-augmentation experiments.
AUGMENTATION_PIPELINE = []


def _elastic_deformation(image, alpha=34, sigma=4):
    h, w = image.shape[:2]
    rng = np.random.RandomState()
    dx = gaussian_filter((rng.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rng.rand(h, w) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = (
        np.clip(y + dy, 0, h - 1).ravel(),
        np.clip(x + dx, 0, w - 1).ravel(),
    )
    out = np.empty_like(image)
    for c in range(image.shape[2]):
        out[..., c] = map_coordinates(image[..., c], indices, order=1).reshape(h, w)
    return out


def _clahe(image):
    img_u8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = [clahe.apply(img_u8[..., c]) for c in range(img_u8.shape[2])]
    return np.stack(channels, axis=-1).astype(np.float32) / 255.0


def _random_gamma(image, lo=0.7, hi=1.5):
    gamma = np.random.uniform(lo, hi)
    return np.clip(image ** gamma, 0.0, 1.0)


def _gaussian_blur(image, sigma_lo=0.5, sigma_hi=1.5):
    sigma = np.random.uniform(sigma_lo, sigma_hi)
    out = np.empty_like(image)
    for c in range(image.shape[2]):
        out[..., c] = gaussian_filter(image[..., c], sigma=sigma)
    return out


def _cutout(image, n_holes=1, min_size=20, max_size=40):
    h, w = image.shape[:2]
    out = image.copy()
    for _ in range(n_holes):
        ph = np.random.randint(min_size, max_size)
        pw = np.random.randint(min_size, max_size)
        y = np.random.randint(0, max(1, h - ph))
        x = np.random.randint(0, max(1, w - pw))
        out[y:y + ph, x:x + pw] = 0.0
    return out


_AUG_FNS = {
    "elastic": _elastic_deformation,
    "clahe":   _clahe,
    "gamma":   _random_gamma,
    "blur":    _gaussian_blur,
    "cutout":  _cutout,
}


def _preprocessing_fn(image):
    for name, fn in _AUG_FNS.items():
        if name in AUGMENTATION_PIPELINE and np.random.random() < 0.5:
            image = fn(image)
    return image


def create_generators():
    use_paper_aug = "paper" in AUGMENTATION_PIPELINE
    has_custom_aug = any(k in AUGMENTATION_PIPELINE for k in _AUG_FNS)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=use_paper_aug,
        vertical_flip=use_paper_aug,
        rotation_range=90 if use_paper_aug else 0,
        preprocessing_function=_preprocessing_fn if has_custom_aug else None,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        f"{DATASET_PATH}/train",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=RANDOM_SEED,
    )
    val_gen = eval_datagen.flow_from_directory(
        f"{DATASET_PATH}/val",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    test_gen = eval_datagen.flow_from_directory(
        f"{DATASET_PATH}/test",
        target_size=IMG_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen
