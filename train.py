import logging
import os
import random
from time import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import TokenClassifierOutput

import config
from feature_embeddings import BertPrep


class BertForNegationCueClassification(BertPreTrainedModel):
    def __init__(self, cnf, n_lexicals=0):
        super().__init__(cnf)
        self.cnf = cnf
        self.num_labels = cnf.num_labels
        self.n_lexicals = n_lexicals

        self.bert = BertModel.from_pretrained(config.PRETRAINED_MODEL)
        self.dropout = nn.Dropout(cnf.hidden_dropout_prob)
        self.classifier = nn.Linear(cnf.hidden_size + self.n_lexicals, cnf.num_labels)

    def forward(
        self,
        input_ids=None,
        lexical_features=None,
        attention_mask=None,
        token_type_ids=None,
        return_dict=None,
        labels=None,
        device=None,
        *args,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.cnf.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        if self.n_lexicals == 0:
            logits = self.classifier(sequence_output)
        else:
            f_tensors = [sequence_output]
            for lex_tensor in lexical_features:
                # lexical_tensor = torch.ones(seq_size[0], seq_size[1], 1) if feature \
                #     else torch.zeros(seq_size[0], seq_size[1], 1)
                f_tensors.append(lex_tensor.to(device))

            logits = self.classifier(torch.cat(tuple(f_tensors), dim=2))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NegCueDataset(Dataset):
    def __init__(self, dataset: dict, n_lexicals: int = 0):
        self.n_lexicals = n_lexicals
        self.encodings = dataset["input_ids"]
        self.mask = dataset["attention_mask"]
        self.labels = dataset["labels"]
        self.lexicals = dataset["lexicals"]
        self.token_ids = dataset["token_ids"]

    def shape_lexicals(self, idx):
        """
        Transpose the axis, so lexicals will have the expected shape
        """
        # lexicals = [torch.transpose(torch.tensor(l, dtype=torch.float), 0, 1) for l in self.lexicals[idx]]
        lexicals = [torch.tensor(l, dtype=torch.float) for l in self.lexicals[idx]]
        return lexicals

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.n_lexicals != 0:
            item = {
                "input_ids": torch.tensor(self.encodings[idx], dtype=torch.long),
                "attention_mask": torch.tensor(self.mask[idx], dtype=torch.long),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "lexicals": self.shape_lexicals(idx),
                "token_ids": torch.tensor(self.token_ids[idx], dtype=torch.long),
            }
        else:
            item = {
                "input_ids": torch.tensor(self.encodings[idx], dtype=torch.long),
                "attention_mask": torch.tensor(self.mask[idx], dtype=torch.long),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "token_ids": torch.tensor(self.token_ids[idx], dtype=torch.long),
            }
        return item


def training_loop(
    model: BertForNegationCueClassification,
    tokenizer: BertTokenizer,
    epochs: int,
    batch_size: int,
    device,
    optimizer,
    scheduler,
    model_name: str,
    train_data_loader: DataLoader,
    dev_data_loader: DataLoader,
    save_every_n_steps: int = 5000,
):

    model.to(device)
    start = time()
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_data_loader):
            batch_time = time()

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Get lexical features if they are present
            lexicals = batch.get("lexicals", None)

            outputs = model(
                input_ids,
                lexicals,
                attention_mask=attention_mask,
                labels=labels,
                device=device,
            )

            loss = outputs.loss
            batch_loss = loss.item()

            # Show training progress
            if step % batch_size == 0 and not step == 0:
                logging.info(
                    f"Batch {step} of {len(train_data_loader)}. Loss: {batch_loss}. "
                    f"Batch time: {time() - batch_time}. Elapsed: {time() - start}."
                )
                model.eval()
                model.train()

            # Create and save model checkpoint
            if step % save_every_n_steps == 0 and not step == 0:
                print("Saving model checkpoint")
                checkpoint_name = f"{model_name}_{epoch}_{step}"
                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": batch_loss,
                    },
                    f"{checkpoint_name}.bin",
                )

                os.makedirs(checkpoint_name, exist_ok=True)

                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(checkpoint_name)
                tokenizer.save_pretrained(checkpoint_name)

            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        total_eval_loss = 0

        for batch in dev_data_loader:
            input_ids = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # Get lexical features if they are present
            lexicals = batch.get("lexicals", None)

            with torch.no_grad():
                outputs = model(
                    input_ids,
                    lexicals,
                    attention_mask=masks,
                    labels=labels,
                    device=device,
                )
                loss = outputs.loss

            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(dev_data_loader)
        print(f"Validation Loss: {avg_val_loss}")

    return model


def train(
    train_data,
    dev_data,
    model_name,
    epochs,
    batch_size,
    seed=None,
    use_lexicals=False,
    bert_model="bert-base-uncased",
    num_warmup_steps=1e2,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    lexicals = ["pos", "possible_prefix", "possible_suffix"] if use_lexicals else []

    train_prep = BertPrep(train_data, lexicals)
    train_dataset = NegCueDataset(
        train_prep.preprocess_dataset(), n_lexicals=train_prep.lexicals_vec_size
    )

    dev_prep = BertPrep(dev_data, lexicals)
    dev_dataset = NegCueDataset(
        dev_prep.preprocess_dataset(), n_lexicals=train_prep.lexicals_vec_size
    )

    logging.info(
        f"Length of train: {len(train_dataset)}, dev: {len(dev_dataset)} sentences."
    )

    # Put data into dataloader
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset)
    )

    dev_data_loader = DataLoader(
        dev_dataset, batch_size=batch_size, sampler=SequentialSampler(dev_dataset)
    )

    tokenizer = BertTokenizer.from_pretrained(bert_model)

    logging.info(
        f"Using lexical features: {lexicals}, encoded features size: {train_prep.lexicals_vec_size}"
    )
    model = BertForNegationCueClassification.from_pretrained(
        bert_model,
        num_labels=len(train_prep.tag2idx),
        n_lexicals=train_prep.lexicals_vec_size,
    )

    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(train_data_loader) * epochs,
    )

    model = training_loop(
        model,
        tokenizer,
        epochs,
        batch_size,
        device,
        optimizer,
        scheduler,
        model_name,
        train_data_loader,
        dev_data_loader,
    )

    logging.info("------------- Training complete! -------------")
    logging.info(f"Saving model checkpoint to: {model_name}")
    os.makedirs(model_name, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)


def train_baseline_model():
    train(
        train_data=config.TRAIN_FEATURES,
        dev_data=config.DEV_FEATURES,
        model_name=config.BSL_MODEL_CKPT,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        use_lexicals=False,
        bert_model=config.PRETRAINED_MODEL,
    )


def train_lexicals_model():
    train(
        train_data=config.TRAIN_FEATURES,
        dev_data=config.DEV_FEATURES,
        model_name=config.LEX_MODEL_CKPT,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        use_lexicals=True,
        bert_model=config.PRETRAINED_MODEL,
    )


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s: %(message)s")
    train_baseline_model()
    train_lexicals_model()
