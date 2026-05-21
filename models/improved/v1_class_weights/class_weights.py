"""
Class weight computation for v1_class_weights.

Sklearn's compute_class_weight produces weights inversely proportional
to class frequency:
    weight_i = total_samples / (n_classes * count_i)

This means underrepresented classes (Glioma, Meningioma) get higher
weights so their misclassifications are penalised more during training.

Dataset counts (Table 1, train split):
    glioma:      1321
    meningioma:  1339
    notumor:     1595
    pituitary:   1457
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(train_generator):
    """
    Computes class weights from the training generator's class distribution.

    Args:
        train_generator: Keras DirectoryIterator (from flow_from_directory)

    Returns:
        dict {class_index: weight} ready to pass to model.fit(class_weight=...)
    """
    class_indices = train_generator.classes   # array of class index per sample
    classes       = np.unique(class_indices)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=class_indices,
    )

    class_weight_dict = dict(zip(classes, weights))

    # Print so it's visible in training output
    idx_to_name = {v: k for k, v in train_generator.class_indices.items()}
    print("\n  Class weights (higher = more penalised when misclassified):")
    print(f"  {'-'*45}")
    for idx, weight in sorted(class_weight_dict.items()):
        print(f"  {idx_to_name.get(idx, idx):<20} index={idx}  weight={weight:.4f}")
    print(f"  {'-'*45}\n")

    return class_weight_dict
