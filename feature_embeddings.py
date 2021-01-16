"""
Words are split on part-word things and the labels of these part words are
duplicated (bert pre-processing)

Creates input_ids, tags, and masks
todo: option that pre-processes lexicals too

Code adapted from
https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
and
https://github.com/abhishekkrthakur/bert-entity-extraction

"""
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import pandas as pd
import torch

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        # todo: add other lexical features (w, t, pos, lemma, stem)
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
    def __init__(self, path, max_sent_len=128): # what would be a good max length
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

        # turn vars to numericals
        all_ids = [self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts]
        all_target_tag = [[self.tag2idx.get(l) for l in lab] for lab in labels]

        # cut/pad the sequences & the bert markers
        for i in range(len(all_ids)):
            # todo: hmm this is just cutting off everything after max length

            # cut until max length, minus 2 to make room for bert markers
            ids = all_ids[i][:self.max_len - 2]
            target_tag = all_target_tag[i][:self.max_len - 2]

            # bert markers and paddings for the rest
            ids = [101] + ids + [102]
            target_tag = [self.tag2idx["PAD"]] + target_tag + [self.tag2idx["PAD"]]

            # pad shortened sequences
            padding_len = self.max_len - len(ids)
            all_ids[i] = ids + ([0] * padding_len)
            all_target_tag[i] = target_tag + ([self.tag2idx["PAD"]] * padding_len)

        # creating the attention mask
        attention_mask = [[float(i != 0.0) for i in ii] for ii in all_ids]

        return {
            "ids": torch.tensor(all_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_tag": torch.tensor(all_target_tag, dtype=torch.long),
            # pos
            # lemma
            # stem
        }

# prep the inputs
train_prep = BertPrep("data/SEM-2012-SharedTask-CD-SCO-training-simple.txt")
train_dataset = train_prep.prep_bert_inputs()

dev_prep = BertPrep("data/SEM-2012-SharedTask-CD-SCO-dev-simple.txt")
dev_dataset = dev_prep.prep_bert_inputs()

# put data into dataloader
bs = 32
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=bs, num_workers=4
)

dev_data_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=bs, num_workers=1
)