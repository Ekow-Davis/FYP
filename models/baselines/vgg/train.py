import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))

from model import build_model, MODEL_NAME, PREPROCESS_FN
from baseline_dataset import create_generators
from baseline_trainer import run_training


def main():
    train_gen, val_gen, _ = create_generators(preprocess_fn=PREPROCESS_FN)
    model = build_model()
    run_training(model, train_gen, val_gen, MODEL_NAME)


if __name__ == "__main__":
    main()
