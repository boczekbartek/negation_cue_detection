# Negation Cue Detection
Code repository for negation cue detection. Project for Applied Text Mining course @ VU

## Description
This repository contains data, code, trained models and results for negation cue detection task.
We present approached based on using pretrained BERT model and finetuning it on SEM-2012 Shared Task dataset for negation cue detection.
We use BERT models similarly to Named Entity Recogniotion task described the in original [paper](https://arxiv.org/abs/1810.04805).
Besides `baseline` model we use the approach of adding POS-tags and pre/suf-fixes to enhance model's performance (`baseline+lexicals` model).

Our repository contains also the code for other lexical features generation as well as annotation study results (`annotations` folder).

## Prerequisities
* Python >= 3.6
* Python libraries:
    * transformers - pretrained BERT model
    * SpaCy - lexical features
    * PyTorch - deep learning backend, data loaders
    * pandas - data processing
    * scikit-learn - evaluation tools
    * numpy - math 
    * nltk - stemmers
    * tqdm - progress bar

All dependencies can be installed with:
```
pip install -r requirements.txt
```

If problems with installation encounter, please visit official libraries' websites.

## Code Usage
### End2end pipeline
Run e2e experiment pipeline including data preprocessing, features generation, baseline and baseline+lexical model trainig and evaluation on devset and testset.
```
python main.py
```

### Features generation
Generate features and store them as `*-features.tsv` inside [data/](data/)` folder. They are already precomputed and stored in this repository.
```
python run_generate_features.py
```

### Training
Train both baseline and baseline+lexicals models.
```
python train.py
```

### Evaluation
Generate error analysis reports and calculate metrics. Results are stored in [reports/](reports/) folder. Our results are included in the repo.
```
python run_evaluate.py
```

## Pre-trained models
We include pre-trained models in the repository with [git-lfs](https://git-lfs.github.com).
* Baseline model: [neg_cue_detection_model_baseline](neg_cue_detection_model_baseline/)
* Baseline+lexicals model: [neg_cue_detection_model_lex](neg_cue_detection_model_lex/)

## Results
All results can be found in `reports/*metrics.txt` files.

## Error Analysis
Although we achieved very good F1 scores our models still make errors. Check them out in `reports/*error_analysis.txt` files.

