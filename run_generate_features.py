import logging

import config
import generate_features


def run_generate_features():
    datasets = [
        config.DEV_DATA,
        config.TRAIN_DATA,
        config.TEST_CARDBOARD_DATA,
        config.TEST_CIRCLE_DATA,
    ]
    fetures_files = [
        config.DEV_FEATURES,
        config.TRAIN_FEATURES,
        config.TEST_CARDBOARD_FEATURES,
        config.TEST_CIRCLE_FEATURES,
    ]

    for d, f in zip(datasets, fetures_files):
        generate_features.run_generate_features(data_file=d, features_file=f)


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s: %(message)s")
    run_generate_features()
