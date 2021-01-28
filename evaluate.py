import logging
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from feature_embeddings import BertPrep
from train import BertForNegationCueClassification, NegCueDataset


def evaluate(ckpt, dataset_file, error_analysis_fname, classification_metrics_fname):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    lexicals = (
        ["pos", "possible_prefix", "possible_suffix"]
        if "_lex" in os.path.basename(ckpt)
        else []
    )
    # TODO this is stupid
    train_prep = BertPrep(
        "data/SEM-2012-SharedTask-CD-SCO-training-simple.v2-features.tsv", lexicals
    )

    n_lexicals = (
        0
        if len(lexicals) == 0
        else sum(
            len(next(iter(v.values()))) for v in train_prep.feature_labels.values()
        )
    )

    logging.info(f"Using lexicals: {lexicals}.")

    model = BertForNegationCueClassification.from_pretrained(
        ckpt, num_labels=len(train_prep.tag2idx), n_lexicals=n_lexicals
    )

    logging.info(f"Running model on device: {device}")
    model.to(device)

    dataset_prep = BertPrep(dataset_file, lexicals)
    dataset = NegCueDataset(dataset_prep.preprocess_dataset(), n_lexicals=n_lexicals)
    dataset_loader = DataLoader(dataset, batch_size=1, num_workers=1)

    true_tags = []
    pred_tags = []

    pred_tags_parsed = []
    true_tags_parsed = []

    tokenizer = train_prep.tokenizer
    inv_tag_enc = {v: k for k, v in train_prep.tag2idx.items()}

    logging.info("Running prediction...")
    with torch.no_grad(), open(error_analysis_fname, "w") as ea_fd:
        for i, data in tqdm(
            enumerate(dataset_loader), total=len(dataset_prep.sentences)
        ):
            sentence = [el[0] for el in dataset_prep.sentences[i]]

            # detect the index of the end of the sentence
            tokenized_sentence = data["input_ids"].numpy().reshape(-1)
            end = [
                i
                for i, tok in enumerate(tokenized_sentence)
                if tok == tokenizer.vocab["[SEP]"]
            ][0]

            # read TRUE labels for this sentence
            true_sent_tags = data["labels"].numpy().reshape(-1)[:end][1:]
            true_tags.extend(true_sent_tags)

            token_ids = data["token_ids"].numpy().reshape(-1)[:end][1:]

            true_tag_parsed = []
            prev_id = None
            for label, tok_id in zip(true_sent_tags, token_ids):
                if prev_id == tok_id:
                    prev_id = tok_id
                    continue
                prev_id = tok_id
                true_tag_parsed.append(label)
            true_tags_parsed.extend(true_tag_parsed)

            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)
            lexicals = None if n_lexicals == 0 else data["lexicals"]

            # Query the model for tags predictions
            outputs = model(
                input_ids,
                lexicals,
                attention_mask=attention_mask,
                labels=labels,
                device=device,
            )
            # Get actual tags from logits and strip them to sentence length
            pred_sent_tags = (
                outputs.logits.argmax(2).cpu().numpy().reshape(-1)[:end][1:]
            )
            pred_tags.extend(pred_sent_tags)

            # Glue tags that were extended because of tokenization.
            pred_tag_parsed = []
            prev_id = None
            for label, tok_id in zip(pred_sent_tags, token_ids):
                if prev_id == tok_id:
                    prev_id = tok_id
                    continue
                prev_id = tok_id
                pred_tag_parsed.append(label)

            pred_tags_parsed.extend(pred_tag_parsed)

            # Print
            if not np.array_equal(true_tag_parsed, pred_tag_parsed):
                print(f"---------------------{i}--------------------------", file=ea_fd)
                print(sentence, file=ea_fd)
                mask = np.array(true_tag_parsed) != np.array(pred_tag_parsed)
                print(np.ma.array(sentence, mask=~mask), file=ea_fd)
                # print('UNK:', [sentence[i] for i in unk_i])
                print("TRUE |", [inv_tag_enc[k] for k in true_tag_parsed], file=ea_fd)
                print("PRED |", [inv_tag_enc[k] for k in pred_tag_parsed], file=ea_fd)
    logging.info(f"Saved data for error analysis to: {error_analysis_fname}")

    with open(classification_metrics_fname, "w") as cr_fd:
        print("--------- Post-processed Classification Report --------- ", file=cr_fd)
        print(
            classification_report(
                true_tags_parsed,
                pred_tags_parsed,
                target_names=list(train_prep.tag2idx.keys()),
                digits=4,
            ),
            file=cr_fd,
        )
        print("--------- Post-processed Confustion Matrix --------- ", file=cr_fd)
        print(inv_tag_enc, file=cr_fd)
        print(confusion_matrix(true_tags, pred_tags), file=cr_fd)
    logging.info(f"Saved classification metrics to: {classification_metrics_fname}")

