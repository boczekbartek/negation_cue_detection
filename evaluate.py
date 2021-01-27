import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from main import NegCueDataset, BertForNegationCueClassification, MODEL_NAME
from feature_embeddings import BertPrep
from tqdm import tqdm
from sklearn.metrics import classification_report

if __name__ == "__main__":    
    checkpoint_name = f"{MODEL_NAME}_{999}_{999}"
    
    lexicals = []
    # lexicals = ["POS", "Possible_Prefix", "Possible_Suffix"]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_prep = BertPrep("data/SEM-2012-SharedTask-CD-SCO-training-simple-v2-features.csv", lexicals)
    n_lexicals = 0 if len(lexicals) == 0 else sum(len(next(iter(v.values()))) for v in train_prep.feature_labels.values())
    train_dataset = NegCueDataset(train_prep.preprocess_dataset(), n_lexicals=n_lexicals)

    dev_prep = BertPrep("data/SEM-2012-SharedTask-CD-SCO-dev-simple-v2-features.csv", lexicals)
    dev_dataset = NegCueDataset(dev_prep.preprocess_dataset(), n_lexicals=n_lexicals)
    
    dev_data_loader = DataLoader(dev_dataset, batch_size=1, num_workers=1)
    
    print(f"Loading checkpoint: {checkpoint_name}")
    model = BertForNegationCueClassification.from_pretrained(
        checkpoint_name,
        num_labels=len(train_prep.tag2idx),
        n_lexicals=n_lexicals
    )
    
    model.to(device)

    true_tags = []
    pred_tags = []
    
    pred_tags_parsed = []
    true_tags_parsed = []
    VERBOSE=False

    tokenizer = train_prep.tokenizer
    inv_tag_enc = {v:k for k,v in train_prep.tag2idx.items()}
    with torch.no_grad():
        for i, data in enumerate(dev_data_loader):
            sentence = [el[0] for el in dev_prep.sentences[i]]
            tok_sentence = dev_prep.tokenized_texts[i]

            # detect the index of the end of the sentence
            tokenized_sentence = data['input_ids'].numpy().reshape(-1)
            end = [i for i, tok in enumerate(tokenized_sentence) if tok == tokenizer.vocab['[SEP]']][0]

            # read TRUE labels for this sentence
            true_sent_tags = data['labels'].numpy().reshape(-1)[: end][1:]
            true_tags.extend(true_sent_tags)

            token_ids = data['token_ids'].numpy().reshape(-1)[: end][1:]

            true_tag_parsed = []
            prev_id = None
            for label, tok_id in zip(true_sent_tags, token_ids):
                if prev_id == tok_id:
                    prev_id = tok_id
                    continue
                prev_id = tok_id
                true_tag_parsed.append(label)
            true_tags_parsed.extend(true_tag_parsed)
            tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence, skip_special_tokens=True)
  
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            lexicals = None if n_lexicals == 0 else data['lexicals']

            # Query the model for tags predictions
            outputs = model(input_ids, lexicals, attention_mask=attention_mask, labels=labels, device=device)            
            # Get actual tags from logits and strip them to sentence length
            pred_sent_tags = outputs.logits.argmax(2).cpu().numpy().reshape(-1)[: end][1:]
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
            
            if not np.array_equal(true_sent_tags, pred_sent_tags):
            # if "n't" in sentence:
            # if '[UNK]' in tokens:
            # unk_i = []
            # for i,t in enumerate(tokens):
            #   if t == '[UNK]':
            #     unk_i.append(i)
              print(f'---------------------{i}--------------------------')
              print(sentence)
              mask=np.array(true_tag_parsed)!=np.array(pred_tag_parsed)
              print(np.ma.array(sentence, mask=~mask))
              # print('UNK:', [sentence[i] for i in unk_i])
              print('TRUE |', [inv_tag_enc[k] for k in true_tag_parsed])
              print('PRED |', [inv_tag_enc[k] for k in pred_tag_parsed])


    print(
      classification_report(true_tags, pred_tags, 
                            target_names=list(train_prep.tag2idx.keys()))
         )
    print(
      classification_report(true_tags_parsed, pred_tags_parsed,
                            target_names=list(train_prep.tag2idx.keys()))
          )