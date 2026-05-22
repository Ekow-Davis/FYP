import os
import shutil
import random

SRC_ROOT = "../data/OCT2026/OCT2026"
DST_ROOT = "../data/oct_retinal_10k"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

SPLIT_MAP = {
    "train": ("train_balanced", 1750),
    "val":   ("val_balanced",   375),
    "test":  ("test_balanced",  375),
}

RANDOM_SEED = 42


def prepare_split(split_name, src_folder, n_per_class):
    for cls in CLASSES:
        src_dir = os.path.join(SRC_ROOT, src_folder, cls)
        dst_dir = os.path.join(DST_ROOT, split_name, cls)
        os.makedirs(dst_dir, exist_ok=True)

        images = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) < n_per_class:
            raise ValueError(
                f"{src_folder}/{cls} has only {len(images)} images, need {n_per_class}"
            )

        selected = random.sample(images, n_per_class)
        for fname in selected:
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

        print(f"  {split_name}/{cls}: copied {n_per_class} images")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    if os.path.exists(DST_ROOT):
        print(f"Destination already exists: {DST_ROOT}")
        print("Delete it manually if you want to re-run.")
        raise SystemExit(1)

    for split_name, (src_folder, n) in SPLIT_MAP.items():
        print(f"\nPreparing {split_name}...")
        prepare_split(split_name, src_folder, n)

    print(f"\nDone. Dataset written to: {DST_ROOT}")
    print("Totals: 7000 train | 1500 val | 1500 test (1750/375/375 per class)")
