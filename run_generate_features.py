import logging
import generate_features


def run_generate_features():
    datasets = [
        "data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt",
        "data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt",
        "data/SEM-2012-SharedTask-CD-SCO-test-cardboard.txt",
        "data/SEM-2012-SharedTask-CD-SCO-test-circle.txt",
    ]
    fetures_files = [".".join(f.split(".")[:-1]) + "-features.tsv" for f in datasets]

    for d, f in zip(datasets, fetures_files):
        generate_features.run_generate_features(data_file=d, features_file=f)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s: %(message)s")
    run_generate_features()
