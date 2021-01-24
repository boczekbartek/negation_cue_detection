import numpy as np
import os
import random
import torch
from time import time
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import TokenClassifierOutput
from feature_embeddings import BertPrep

EPOCHS = 100
BATCH_SIZE = 8
SEED = 777
MODEL_NAME = 'neg_cue_detection_model'


class BertForNegationCueClassification(BertPreTrainedModel):
    def __init__(self, config, n_lexicals):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + n_lexicals,
                                    config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, lexical_features=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

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
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
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
    def __init__(self, dataset: dict):
        self.encodings = dataset['input_ids']
        self.mask = dataset['attention_mask']
        self.labels = dataset['labels']
        self.lexicals = dataset['lexicals']

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
        item = {'input_ids': torch.tensor(self.encodings[idx], dtype=torch.long),
                'attention_mask': torch.tensor(self.mask[idx], dtype=torch.long),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long),
                'lexicals': self.shape_lexicals(idx)}
        return item


if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # choose lexical features
    lexicals = ["POS", "Possible_Prefix", "Possible_Suffix"]

    # Prep the inputs
    print("Preprocessing data")
    train_prep = BertPrep("data/SEM-2012-SharedTask-CD-SCO-training-simple-v2-features.csv", lexicals)
    train_dataset = NegCueDataset(train_prep.preprocess_dataset())

    dev_prep = BertPrep("data/SEM-2012-SharedTask-CD-SCO-dev-simple-v2-features.csv", lexicals)
    dev_dataset = NegCueDataset(dev_prep.preprocess_dataset())

    # Put data into dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(train_dataset))
    dev_data_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, sampler=SequentialSampler(dev_dataset))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNegationCueClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(train_prep.tag2idx),
        n_lexicals=sum(len(next(iter(v.values()))) for v in train_prep.feature_labels.values())
    )

    model.to(device)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1e2,
                                                num_training_steps=len(train_data_loader) * EPOCHS)

    start = time()
    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_data_loader):
            batch_time = time()

            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            lexicals = batch['lexicals']
            outputs = model(input_ids, lexicals, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            batch_loss = loss.item()

            # Show training progress
            if step % BATCH_SIZE == 0 and not step == 0:
                print(f"Batch {step} of {len(train_data_loader)}. Loss: {batch_loss}. "
                      f"Batch time: {time() - batch_time}. Elapsed: {time() - start}.")
                model.eval()
                model.train()

            # Create and save model checkpoint
            if step % 5000 == 0 and not step == 0:
                print('Saving model checkpoint')
                checkpoint_name = f"{MODEL_NAME}_{epoch}_{step}"
                torch.save({'epoch': epoch,
                            'step': step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': batch_loss}, f"{checkpoint_name}.bin")

                if not os.path.exists(checkpoint_name):
                    os.makedirs(checkpoint_name)

                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(checkpoint_name)
                tokenizer.save_pretrained(checkpoint_name)

            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        total_eval_loss = 0

        for batch in dev_data_loader:
            input_ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            lexicals = batch['lexicals']

            with torch.no_grad():
                outputs = model(input_ids, lexicals, attention_mask=masks, labels=labels)
                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(dev_data_loader)
        print(f"Validation Loss: {avg_val_loss}")

    print("\n\n ------------- \n Training complete! \n  -------------")
    print('Saving model checkpoint')
    checkpoint_name = f"{MODEL_NAME}_{999}_{999}"
    if not os.path.exists(checkpoint_name):
        os.makedirs(checkpoint_name)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(checkpoint_name)
    tokenizer.save_pretrained(checkpoint_name)
