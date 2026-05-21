"""
v2_gradcam — Grad-CAM heatmap generator.

Works with ANY saved LEAD-CNN .keras model (lead_cnn, v1, v3, v4).
No training involved — purely post-training analysis.

Grad-CAM explanation:
  Computes the gradient of the predicted class score with respect to
  the final conv layer's feature maps. Regions with large positive
  gradients contributed most to that prediction — shown as red/warm
  areas in the heatmap overlay.

Usage:
  python gradcam.py
      --model  path/to/model.keras
      --images path/to/test/images/     (folder or single image)
      --output path/to/save/heatmaps/
      --layer  conv6                    (optional, defaults to conv6)
      --n      20                       (number of images to process)

Output per image:
  original.png   — original MRI
  heatmap.png    — Grad-CAM heatmap only
  overlay.png    — heatmap blended onto original (most useful)
  metadata.json  — predicted class, confidence, true class if available
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import cv2

_HERE     = os.path.dirname(os.path.abspath(__file__))
_SHARED   = os.path.join(_HERE, "..", "shared")
_LEAD_CNN = os.path.join(_HERE, "..", "..", "lead_cnn")

sys.path.insert(0, _SHARED)
sys.path.insert(0, _LEAD_CNN)

from config import IMG_SIZE, CLASS_NAMES
from improved_config import IMPROVED_RESULTS_DIR

VARIANT_NAME    = "v2_gradcam"
DEFAULT_LAYER   = "conv6"           # last backbone conv before dim reduction block
DEFAULT_OUT_DIR = os.path.join(IMPROVED_RESULTS_DIR, VARIANT_NAME)


# ── Core Grad-CAM ─────────────────────────────────────────────────────────────

def make_gradcam_heatmap(img_array, model, layer_name, pred_index=None):
    """
    Computes a Grad-CAM heatmap for img_array using the specified layer.

    Args:
        img_array:   Preprocessed image, shape (1, H, W, 3), values in [0,1]
        model:       Loaded Keras model
        layer_name:  Name of the convolutional layer to target
        pred_index:  Class index to explain. If None, uses the top prediction.

    Returns:
        heatmap: numpy array shape (H, W), values in [0, 1]
        pred_index: the class that was explained
        confidence: softmax probability for that class
    """
    # Build a sub-model that outputs both the target layer and the final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        layer_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of class score w.r.t. feature map
    grads = tape.gradient(class_channel, layer_output)

    # Mean gradient over spatial dimensions -> importance weight per filter
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by importance and collapse to single channel
    layer_output = layer_output[0]
    heatmap = layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalise to [0, 1], clamp negatives
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    confidence = float(predictions[0][pred_index])
    return heatmap, int(pred_index), confidence


def overlay_heatmap(heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Resizes heatmap to match original_img and blends them.

    Args:
        heatmap:      (H, W) numpy array, values in [0, 1]
        original_img: (H, W, 3) numpy array, uint8 [0, 255]
        alpha:        Heatmap opacity (0=invisible, 1=full heatmap)

    Returns:
        overlay: (H, W, 3) numpy array, uint8
    """
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (1 - alpha) * original_img + alpha * heatmap_colored
    return np.uint8(np.clip(overlay, 0, 255))


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(image_path, img_size):
    """Loads and preprocesses a single image for model inference."""
    img = tf.keras.utils.load_img(image_path, target_size=img_size[:2])
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return img_array, np.uint8(img_array * 255)


def collect_images(images_path, n):
    """
    Collects up to n image paths from a file or directory.
    If directory, samples n images across all class subdirectories evenly.
    """
    images_path = os.path.abspath(images_path)
    if os.path.isfile(images_path):
        return [(images_path, None)]

    # Directory — walk class subdirs
    supported = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = []
    for root, _, files in os.walk(images_path):
        class_name = os.path.basename(root)
        for fname in files:
            if os.path.splitext(fname)[1].lower() in supported:
                all_images.append((os.path.join(root, fname), class_name))

    # Sample evenly across classes
    from collections import defaultdict
    by_class = defaultdict(list)
    for path, cls in all_images:
        by_class[cls].append((path, cls))

    per_class = max(1, n // len(by_class)) if by_class else n
    sampled   = []
    for cls_images in by_class.values():
        sampled.extend(cls_images[:per_class])

    return sampled[:n]


# ── Main processing loop ──────────────────────────────────────────────────────

def run_gradcam(model_path, images_path, output_dir, layer_name, n):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Verify target layer exists
    layer_names = [l.name for l in model.layers]
    if layer_name not in layer_names:
        print(f"  Layer '{layer_name}' not found. Available conv layers:")
        for name in layer_names:
            if 'conv' in name.lower():
                print(f"    {name}")
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    image_list = collect_images(images_path, n)
    print(f"Processing {len(image_list)} images → {output_dir}\n")

    summary = []

    for i, (img_path, true_class) in enumerate(image_list):
        img_array, img_uint8 = load_image(img_path, IMG_SIZE)
        img_batch = np.expand_dims(img_array, axis=0)

        heatmap, pred_index, confidence = make_gradcam_heatmap(
            img_batch, model, layer_name
        )

        pred_class = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else str(pred_index)
        correct    = (pred_class == true_class) if true_class else None

        # Save outputs
        img_stem  = os.path.splitext(os.path.basename(img_path))[0]
        safe_stem = f"{i:04d}_{img_stem}"
        img_out   = os.path.join(output_dir, safe_stem)
        os.makedirs(img_out, exist_ok=True)

        # Original
        cv2.imwrite(
            os.path.join(img_out, "original.png"),
            cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        )

        # Heatmap only
        heatmap_vis = cv2.resize(heatmap, IMG_SIZE[:2][::-1])
        cv2.imwrite(
            os.path.join(img_out, "heatmap.png"),
            cv2.applyColorMap(np.uint8(255 * heatmap_vis), cv2.COLORMAP_JET)
        )

        # Overlay (most useful)
        overlay = overlay_heatmap(heatmap, img_uint8)
        cv2.imwrite(
            os.path.join(img_out, "overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        # Metadata
        meta = {
            "image":       img_path,
            "true_class":  true_class,
            "pred_class":  pred_class,
            "pred_index":  pred_index,
            "confidence":  round(confidence, 4),
            "correct":     correct,
            "layer_used":  layer_name,
        }
        with open(os.path.join(img_out, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        status = "✓" if correct else ("✗" if correct is False else "?")
        print(f"  [{i+1:>3}/{len(image_list)}] {status}  "
              f"true={true_class or '?':<12} "
              f"pred={pred_class:<12} "
              f"conf={confidence:.3f}  → {img_out}")

        summary.append(meta)

    # Save run summary
    summary_path = os.path.join(output_dir, "gradcam_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    correct_count = sum(1 for m in summary if m['correct'] is True)
    total_labelled = sum(1 for m in summary if m['correct'] is not None)

    print(f"\n  Done. {len(summary)} images processed.")
    if total_labelled:
        print(f"  Accuracy on this sample: "
              f"{correct_count}/{total_labelled} = "
              f"{100*correct_count/total_labelled:.1f}%")
    print(f"  Summary saved: {summary_path}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for LEAD-CNN predictions."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to saved .keras model file.",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Path to image file or directory of images (with class subfolders).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUT_DIR,
        help=f"Directory to save heatmaps. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--layer",
        default=DEFAULT_LAYER,
        help=f"Target conv layer name. Default: {DEFAULT_LAYER}",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of images to process. Default: 20",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gradcam(
        model_path=args.model,
        images_path=args.images,
        output_dir=args.output,
        layer_name=args.layer,
        n=args.n,
    )
