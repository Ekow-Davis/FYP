import os
import shutil
import random

SEED = 42
KVASIR_ROOT = "../data/KvasirV2"
OUTPUT_ROOT = "../data/gi_data"

CLASSES = ["esophagitis", "polyps", "ulcerative-colitis", "normal-pylorus"]
SPLITS = {"train": 800, "val": 100, "test": 100}


def prepare():
    random.seed(SEED)

    for cls in CLASSES:
        src_dir = os.path.join(KVASIR_ROOT, cls, cls)
        images = sorted([
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(images) < sum(SPLITS.values()):
            raise ValueError(f"{cls}: only {len(images)} images, need {sum(SPLITS.values())}")

        random.shuffle(images)

        offset = 0
        for split, count in SPLITS.items():
            split_images = images[offset:offset + count]
            offset += count
            dest_dir = os.path.join(OUTPUT_ROOT, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for img in split_images:
                shutil.copy2(os.path.join(src_dir, img), os.path.join(dest_dir, img))

        print(f"{cls}: {SPLITS}")

    print("\nDone. gi_data split complete.")


if __name__ == "__main__":
    prepare()
