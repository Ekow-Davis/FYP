import os
import re
from collections import defaultdict

SRC_ROOT = "../data/OCT2026/OCT2026"
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
TARGET_IMAGES_PER_CLASS = 3500

def parse_patient_id(filename):
    """Extract patient ID from filename like 'CNV-81630-5.jpeg'"""
    match = re.match(r'[A-Z]+-(\d+)-\d+', filename)
    return match.group(1) if match else None

def check_patient_feasibility():
    results = {}

    for split in ["train_balanced", "val_balanced", "test_balanced"]:
        results[split] = {}
        for cls in CLASSES:
            cls_dir = os.path.join(SRC_ROOT, split, cls)

            patient_to_images = defaultdict(list)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    patient_id = parse_patient_id(fname)
                    if patient_id:
                        patient_to_images[patient_id].append(fname)

            unique_patients = len(patient_to_images)
            total_images = sum(len(imgs) for imgs in patient_to_images.values())
            avg_images_per_patient = total_images / unique_patients if unique_patients > 0 else 0

            results[split][cls] = {
                "total_images": total_images,
                "unique_patients": unique_patients,
                "avg_images_per_patient": avg_images_per_patient,
            }

            print(f"{split}/{cls}:")
            print(f"  Total images: {total_images}")
            print(f"  Unique patients: {unique_patients}")
            print(f"  Avg images/patient: {avg_images_per_patient:.2f}")

    # Check feasibility for train split (most important)
    print("\n" + "="*60)
    print("FEASIBILITY CHECK FOR PATIENT-LEVEL SPLIT (3500 images/class)")
    print("="*60)

    train_results = results["train_balanced"]
    feasible = True
    min_patients = float('inf')

    for cls in CLASSES:
        needed_patients = TARGET_IMAGES_PER_CLASS / train_results[cls]["avg_images_per_patient"]
        available_patients = train_results[cls]["unique_patients"]
        is_feasible = available_patients >= needed_patients
        feasible = feasible and is_feasible
        min_patients = min(min_patients, available_patients)

        status = "OK" if is_feasible else "INSUFFICIENT"
        print(f"{cls}: need {needed_patients:.0f} patients, have {available_patients} {status}")

    print("\n" + "="*60)
    if feasible:
        print("RESULT: Patient-level split IS FEASIBLE")
        print("Use Option B: Prepare with patient-level deduplication")
        return {"feasible": True, "strategy": "patient_level"}
    else:
        print("RESULT: Patient-level split NOT FEASIBLE")
        print(f"Bottleneck: class with only {min_patients} patients")
        print("Fallback: Use Option A (random sampling, accept patient leakage)")
        return {"feasible": False, "strategy": "random_sampling"}

if __name__ == "__main__":
    result = check_patient_feasibility()
    print(f"\nRecommended strategy: {result['strategy'].upper()}")
