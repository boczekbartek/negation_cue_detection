"""
Words are split on part-word things and the labels of these part words are
duplicated.

In the tutorial they use data with pos tags, but these tags are not included
in the eventual input_ids, tags, or masks

code adapted from
https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
"""
from transformers import BertForTokenClassification, BertTokenizer, BertModel
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import torch

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["token"].values.tolist(),
                                                     s["tag"].values.tolist())]
        self.grouped = self.data.groupby("n_sent").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# prepare sentences and labels for bert
class BertPrep():
    def __init__(self, path, max_sent_len=75):
        # chose smallest pre-trained bert (uncased)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data = self.load_data(path)
        self.tag2idx = self.create_tag_dict()
        self.max_len = max_sent_len

    def load_data(self, path):
        data = pd.read_csv(path,
                           delimiter="\t",
                           names=["corpus", "n_sent", "n_word", "token", "tag"])
        return data

    def create_tag_dict(self):
        # convert the tags to indices and create dict for conversion
        tag_values = list(set(self.data["tag"].values))
        tag_values.append("PAD")
        tag2idx = {t: i for i, t in enumerate(tag_values)}
        return tag2idx

    def tokenize_and_preserve_labels(self, sentence, text_labels):
        """
        tokenizes the words into the bert word elements and duplicates the labels
        :return:
        """
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels


    def prep_bert_inputs(self):
        # get the sentences and the labels from the dataframe
        getter = SentenceGetter(self.data)
        sentences = [[s[0] for s in sent] for sent in getter.sentences]
        labels = [[s[1] for s in sent] for sent in getter.sentences]

        # tokenize the words + duplicate the labels --> list of tuples with (sent tokens, labels)
        tokenized_texts_and_labels = [
            self.tokenize_and_preserve_labels(sent, labs)
            for sent, labs in zip(sentences, labels)
        ]

        # create seperate vars for the tokenized text and their labels
        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
        labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

        # cut/pad the sequences & the bert markers
        all_ids = [self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts]
        marked_ids = [[101] + ids + [102] for ids in all_ids]
        input_ids = pad_sequences(marked_ids,
                                  maxlen=self.max_len, dtype="long", value=0.0,
                                  truncating="post", padding="post")

        all_tags = [[self.tag2idx.get(l) for l in lab] for lab in labels]
        marked_tags = [[0] + ids + [0] for ids in all_tags]
        tags = pad_sequences(marked_tags,
                             maxlen=self.max_len, value=self.tag2idx["PAD"], padding="post",
                             dtype="long", truncating="post")

        # todo: add for other lexical things too

        # creating the attention mask
        attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

        # todo: turn to tensors

        return input_ids, tags, attention_masks

# prep the inputs
train_prep = BertPrep("data/SEM-2012-SharedTask-CD-SCO-training-simple.txt")
tr_inputs, tr_tags, tr_masks = train_prep.prep_bert_inputs()

dev_prep = BertPrep("data/SEM-2012-SharedTask-CD-SCO-dev-simple.txt")
dev_inputs, dev_tags, dev_masks = dev_prep.prep_bert_inputs()

# turn inputs to tensors
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(dev_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(dev_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(dev_masks)

# put data into dataloader
bs = 32
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

# todo: OR add POS tags to the input input_ids OR extra input per instance aside from embedding??
