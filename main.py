import os
import torch
from time import time
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForTokenClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from feature_embeddings import BertPrep

EPOCHS = 3
BATCH_SIZE = 32
MODEL_NAME = 'neg_cue_detection_model'


class NegCueDataset(Dataset):
    def __init__(self, dataset: dict):
        self.encodings = dataset['input_ids']
        self.mask = dataset['attention_mask']
        self.labels = dataset['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {'input_ids': torch.tensor(self.encodings[idx], dtype=torch.long),
                'attention_mask': torch.tensor(self.mask[idx], dtype=torch.long),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)}
        return item


if __name__ == "__main__":
    # Prep the inputs
    train_prep = BertPrep("data/SEM-2012-SharedTask-CD-SCO-training-simple.txt")
    train_dataset = NegCueDataset(train_prep.preprocess_dataset())

    dev_prep = BertPrep("data/SEM-2012-SharedTask-CD-SCO-dev-simple.txt")
    dev_dataset = NegCueDataset(dev_prep.preprocess_dataset())

    # Put data into dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(train_dataset))
    dev_data_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, sampler=SequentialSampler(dev_dataset))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(train_prep.tag2idx))
    model.to(device)

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
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

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

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=masks, labels=labels)
                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(dev_data_loader)
        print(f"Validation Loss: {avg_val_loss}")

    print("\n\n ------------- \n Training complete! \n  -------------")
