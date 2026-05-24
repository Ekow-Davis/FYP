import os
import shutil
import random
import re
from collections import defaultdict

SRC_ROOT = "../data/OCT2026/OCT2026"
DST_ROOT = "../data/oct_retinal_10k"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

SPLIT_MAP = {
    "train": ("train_balanced", 3500),
    "val":   ("val_balanced",   750),
    "test":  ("test_balanced",  750),
}

RANDOM_SEED = 42


def parse_patient_id(filename):
    """Extract patient ID from filename like 'CNV-81630-5.jpeg'"""
    match = re.match(r'[A-Z]+-(\d+)-\d+', filename)
    return match.group(1) if match else None


def prepare_split(split_name, src_folder, n_per_class):
    for cls in CLASSES:
        src_dir = os.path.join(SRC_ROOT, src_folder, cls)
        dst_dir = os.path.join(DST_ROOT, split_name, cls)
        os.makedirs(dst_dir, exist_ok=True)

        # Group images by patient to avoid leakage
        patient_to_images = defaultdict(list)
        for fname in os.listdir(src_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                patient_id = parse_patient_id(fname)
                if patient_id:
                    patient_to_images[patient_id].append(fname)

        # Randomly select patients until we reach target image count
        patients = list(patient_to_images.keys())
        random.shuffle(patients)

        selected_images = []
        for patient_id in patients:
            selected_images.extend(patient_to_images[patient_id])
            if len(selected_images) >= n_per_class:
                selected_images = selected_images[:n_per_class]
                break

        # Copy selected images
        for fname in selected_images:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            shutil.copy2(src_path, dst_path)

        print(f"  {split_name}/{cls}: copied {len(selected_images)} images")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    if os.path.exists(DST_ROOT):
        print(f"Removing existing dataset: {DST_ROOT}")
        shutil.rmtree(DST_ROOT)

    for split_name, (src_folder, n) in SPLIT_MAP.items():
        print(f"\nPreparing {split_name}...")
        prepare_split(split_name, src_folder, n)

    print(f"\nDone. Dataset written to: {DST_ROOT}")
    print("Totals: 14000 train | 3000 val | 3000 test (3500/750/750 per class)")
    print("Patient-level deduplication applied (no patient leakage)")

