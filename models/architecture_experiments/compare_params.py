"""
compare_params.py

Builds every architecture experiment model and prints a parameter
comparison table against the base LEAD-CNN.

Run from the project root:
    python models/architecture_experiments/compare_params.py

No training required — just builds each model and counts parameters.
"""

import os
import sys

_HERE     = os.path.dirname(os.path.abspath(__file__))
_SHARED   = os.path.join(_HERE, "shared")
_LEAD_CNN = os.path.join(_HERE, "..", "lead_cnn")

sys.path.insert(0, _SHARED)
sys.path.insert(0, _LEAD_CNN)

import importlib.util
import tensorflow as tf

# Suppress TF logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def load_config(experiment_dir, config_name):
    spec = importlib.util.spec_from_file_location(
        config_name,
        os.path.join(experiment_dir, "config.py")
    )
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def get_param_count(model):
    total     = model.count_params()
    trainable = sum(l.count_params() for l in model.layers if l.trainable)
    return total, trainable


# ── Build each model ──────────────────────────────────────────────────────────

results = []

# Base LEAD-CNN
try:
    from architecture import build_lead_cnn
    m = build_lead_cnn()
    total, trainable = get_param_count(m)
    results.append(("Base LEAD-CNN (paper)", total, trainable, 0))
    BASE_PARAMS = total
    tf.keras.backend.clear_session()
except Exception as e:
    print(f"Base LEAD-CNN: failed ({e})")
    BASE_PARAMS = 1132612
    results.append(("Base LEAD-CNN (paper)", BASE_PARAMS, BASE_PARAMS, 0))

# Architecture experiments
experiments = [
    ("depthwise",           "depthwise_config",           "architecture", "build_dsc_lead_cnn"),
    ("attention",           "attention_config",            "architecture", "build_se_lead_cnn"),
    ("combined",            "combined_config",             "architecture", "build_combined_lead_cnn"),
    ("dsc_dimred",          "dsc_dimred_config",           "architecture", "build_dsc_dimred_lead_cnn"),
    ("dsc_dimred_attention","dsc_dimred_attention_config",  "architecture", "build_dsc_dimred_attention_lead_cnn"),
]

for exp_name, cfg_name, arch_module, build_fn in experiments:
    exp_dir = os.path.join(_HERE, exp_name)
    if not os.path.exists(exp_dir):
        continue

    try:
        # Load config
        cfg = load_config(exp_dir, cfg_name)

        # Load architecture module
        arch_spec = importlib.util.spec_from_file_location(
            f"{exp_name}_arch",
            os.path.join(exp_dir, "architecture.py")
        )
        arch_mod = importlib.util.module_from_spec(arch_spec)
        arch_spec.loader.exec_module(arch_mod)

        # Build model
        build = getattr(arch_mod, build_fn)
        model = build(config=cfg)
        total, trainable = get_param_count(model)
        results.append((exp_name, total, trainable, total - BASE_PARAMS))
        tf.keras.backend.clear_session()

    except Exception as e:
        results.append((exp_name, None, None, None))
        print(f"  Warning: {exp_name} failed to build: {e}")

# ── Print table ───────────────────────────────────────────────────────────────

sep = "=" * 78
print(f"\n{sep}")
print(f"  Architecture Parameter Comparison")
print(sep)
print(f"  {'Model':<35} {'Total Params':>14} {'Trainable':>12} {'vs Base':>12}")
print(f"  {'-'*35} {'-'*14} {'-'*12} {'-'*12}")

for name, total, trainable, diff in results:
    if total is None:
        print(f"  {name:<35} {'BUILD FAILED':>14}")
        continue
    diff_str = f"{diff:+,}" if diff != 0 else "—"
    print(f"  {name:<35} {total:>14,} {trainable:>12,} {diff_str:>12}")

print(sep)
print(f"\n  Note: 'vs Base' shows difference from Base LEAD-CNN ({BASE_PARAMS:,} params)")
print(f"  Negative = fewer parameters (more lightweight)")
print(f"  Positive = more parameters\n")
