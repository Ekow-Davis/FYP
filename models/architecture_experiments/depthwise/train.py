"""
Depthwise experiment — training script.
Modify config.py to change hyperparameters, then rerun this file.
"""

import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.join(_HERE, "..", "shared")
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)

import importlib.util

# Load config by path to avoid name collisions
_spec = importlib.util.spec_from_file_location(
    "depthwise_config", os.path.join(_HERE, "config.py")
)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

from architecture import build_dsc_lead_cnn
from arch_dataset import create_generators
from arch_trainer import run_training


def main():
    train_gen, val_gen, _ = create_generators(batch_size=config.BATCH_SIZE)
    model = build_dsc_lead_cnn(config=config)
    run_training(model, train_gen, val_gen, config)


if __name__ == "__main__":
    main()
