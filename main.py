""" Run negation cue detection experiment end-2-end """
import logging

from run_evaluate import run_evaluate
from run_generate_features import run_generate_features
from train import train_baseline_model, train_lexicals_model


def main():
    run_generate_features()
    train_baseline_model()
    train_lexicals_model()
    run_evaluate()


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s: %(message)s")
    main()
