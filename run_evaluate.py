""" Evaluate trained models and print errors on test and dev datasets """
import logging
import os
from os.path import basename

from evaluate import evaluate


def run_evaluate():
    """ 
    Run evaluation on 2 models:
        * baseline
        * baseline + lexical
    Save evaluation and error analysis data to reports directory.
    """
    checkpoints = [
        "neg_cue_detection_model_baseline",
        "neg_cue_detection_model_lex",
    ]
    datasets = [
        "data/SEM-2012-SharedTask-CD-SCO-dev-simple-v2-features.csv",
        "data/SEM-2012-SharedTask-CD-SCO-test-circle-features.csv",
        "data/SEM-2012-SharedTask-CD-SCO-test-cardboard-features.csv",
    ]
    reports_dir = "reports"
    logging.info(f"Reports directory: {reports_dir}")
    os.makedirs(reports_dir, exist_ok=True)

    for ckpt in checkpoints:
        for dataset in datasets:
            lex = "_lex_" in ckpt
            error_analysis_file = f'{reports_dir}/{basename(dataset)}.{"lex." if lex else ""}error_analysis.txt'
            metrics_file = (
                f'{reports_dir}/{basename(dataset)}.{"lex." if lex else ""}metrics.txt'
            )
            logging.info(
                f"\nModel: {ckpt}\nDataset: {dataset}\n-> Error Analysis: {error_analysis_file}\n-> Metrics: {metrics_file}"
            )

            evaluate(
                ckpt=ckpt,
                dataset_file=dataset,
                error_analysis_fname=error_analysis_file,
                classification_metrics_fname=metrics_file,
            )


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s: %(message)s")
    run_evaluate()

