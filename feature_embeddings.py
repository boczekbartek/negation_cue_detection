"""
Words are split on part-word things and the labels of these part words are
duplicated (bert pre-processing)

Creates input_ids, tags, and masks
Also pre-processes lexical features if included

Code adapted from
https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
and
https://github.com/abhishekkrthakur/bert-entity-extraction

"""
from generate_features import generate_features
from transformers import BertTokenizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


POS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CONJ",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    "SPACE",
]


class SentenceGetter(object):
    def __init__(self, data, lexicals=None):
        self.n_sent = 1
        self.data = data
        self.empty = False

        agg_func = lambda s: self.aggregate(s, lexicals)

        self.grouped = self.data.groupby("sent_id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    @staticmethod
    def aggregate(s, lexicals: list = None):
        """
        Groups the Token, Tag, and lexical features per word in a sentence.

        :return: the sentence as a list with a tuple per word
                 [(token, tag, feat1, feat2), (..), ...]
        """
        sentence_feat = []

        # which features to extract
        to_extract = [s["token_lower"].values.tolist(), s["tag"].values.tolist()]

        if lexicals:
            to_extract.extend(s[lexical].values.tolist() for lexical in lexicals)

        # retrieve the information
        for zippies in zip(*to_extract):
            sentence_feat.append(zippies)

        return sentence_feat

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# prepare sentences and labels for bert
class BertPrep(object):
    def __init__(self, path, lexicals=None, max_sent_len=95):
        # chose smallest pre-trained bert (uncased)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.data = self.load_data(path)
        self.tag2idx = {"B-NEG": 0, "O": 1, "I-NEG": 2}
        self.feature_labels = self.label_lexicals(lexicals)
        self.max_len = max_sent_len
        self.lexicals = lexicals
        self.lexicals_vec_size = self.calculate_lexicals_size()

    def calculate_lexicals_size(self):
        if len(self.lexicals) == 0:
            return 0
        else:
            return sum(
                len(next(iter(v.values()))) for v in self.feature_labels.values()
            )

    @staticmethod
    def load_data(path):
        # check if the file contains the lexical features
        ext = path.split(".")[1]

        # generate the features if the current file doesn't have lexicals
        if ext == "txt":
            print("generating lexical features")
            data = pd.read_csv(
                path,
                delimiter="\t",
                names=["corpus", "n_sent", "n_word", "token", "tag"],
            )
            data = generate_features(data)
            data.to_csv(path.split(".")[0] + "-features.csv", sep="\t")
            print(
                f"new file with generated features saved as: "
                f'{path.split(".")[0] + "-features.csv"}'
            )

        else:
            # features are included in the csv :')
            data = pd.read_csv(path, delimiter="\t", header=0, index_col=0)
            data["sent_id"] = data["corpus"] + data["n_sent"].astype(str)

        return data

    def set_max_len(self, tokenized_texts):
        """
        Base max length of padding sequence on longest tokenized text
        (word pieces)
        """
        self.max_len = max([len(sent) for sent in tokenized_texts])

    def create_label_dict(self, feature, scale: bool):
        """
        Convert the tags to indices and create dict for conversion.

        :return:
        """
        values = list(set(self.data[feature].values))
        if feature != "tag":
            values.insert(0, "PAD")

        if scale:
            step = np.linspace(0, 1, len(values))
            feat2idx = {t: step[i] for i, t in enumerate(values)}
        else:
            feat2idx = {t: i for i, t in enumerate(values)}
        return feat2idx

    def label_lexicals(self, lexicals) -> dict:
        """
        Create value mapping for all features
        Scales the labels

        :return: dict of lexicals, containing dicts of value to numericals
        """

        def encode(feat):
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
            if feat == "possible_prefix" or feat == "possible_suffix":
                v = [True, False]
                one_hot = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
            else:
                v = np.array(POS_TAGS)
                one_hot = enc.fit_transform(v.reshape(len(v), 1))

            d = {label: one_hot[i] for i, label in enumerate(v)}
            d["PAD"] = np.zeros(one_hot.shape[1])
            return d

        if lexicals:
            return {feat: encode(feat) for feat in lexicals}

    def tokenize_and_duplicate_labels(self, aggreg_sentence):
        """
        Tokenizes the words of one sentence into the bert word pieces,
        duplicates the labels and the lexical word features

        :return:
        """
        # hhhh house of inefficient code only, sorry
        tokenized_sentence = []
        labels = []
        lexicals = None
        token_ids = []
        if self.lexicals:
            lexicals = {lexical: [] for lexical in self.lexicals}

        for i, aggreg_word in enumerate(aggreg_sentence):
            word_features = list(aggreg_word)

            # pop the word and tag
            word = word_features.pop(0)
            label = word_features.pop(0)

            # process word: tokenize and count how many subwords it's broken into
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_sentence.extend(tokenized_word)

            # duplicate labels
            labels.extend(self.duplicate(label, n_subwords))
            token_ids.extend(self.duplicate(i, n_subwords))

            if self.lexicals:
                # duplicate lexicals
                for i, lex in enumerate(self.lexicals):
                    feat = word_features[i]
                    lexicals[lex].extend(self.duplicate(feat, n_subwords))

        return tokenized_sentence, labels, lexicals, token_ids

    @staticmethod
    def duplicate(label, n_subwords):
        """
        Duplicates a label or feature element
        """
        duplicates = [label] * n_subwords
        return duplicates

    def encode_lexicals(self, lexicals):
        """
        (Nested loop of hell, i know, im sorry)
        BUT basically, loops to the lexical features per sentence, then encodes
        them according to the dicts defined in self.feature_labels

        :param lexicals: a list (all sentences)
                         of dicts (one sentence)
                         of key (lexical features): value (list of lexical per word)

        :return: a list of dicts with a list, but now with numbers~
        """
        all_lexicals = [
            {
                lex: [self.feature_labels[lex].get(feat) for feat in sent[lex]]
                for lex in sent.keys()
            }
            for sent in lexicals
        ]
        return all_lexicals

    @staticmethod
    def pad_start_end(encoder, feature_to_pad):
        """
        Adds padding to features to align with the [cls] and [sep] markers.
        :param encoder: the label to index dictionary
        :param feature_to_pad:
        :return:
        """
        if "O" in encoder:
            return [encoder["O"]] + feature_to_pad + [encoder["O"]]
        else:
            return [encoder["PAD"]] + feature_to_pad + [encoder["PAD"]]

    def preprocess_dataset(self):
        """
        1) Chops words up into Bert word pieces, duplicates the labels and lexical
           features with it.
        2) Turns variables into numericals
        3) Cut and pad the sequences to length self.max_length
        4) Add Bert cls and sep sentence markers
        5) Create attention mask

        :return: dict of input_ids, mask, labels, lexicals
        """

        # get the sentences and the labels from the dataframe
        getter = SentenceGetter(self.data, self.lexicals)
        self.sentences = getter.sentences
        # tokenize the words + duplicate the labels and lexical features
        tokenized_texts, tags, lexicals, token_ids = map(
            list,
            zip(
                *[
                    self.tokenize_and_duplicate_labels(sentence)
                    for sentence in getter.sentences
                ]
            ),
        )

        self.tokenized_texts = tokenized_texts
        # turn vars to numericals
        all_ids = [
            self.tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_texts
        ]
        all_target_tags = [[self.tag2idx.get(tag) for tag in sent] for sent in tags]

        # don't break when no lexical features are included pls
        all_lexicals = None
        if self.lexicals:
            all_lexicals = self.encode_lexicals(lexicals)
            processed_lex = []
        else:
            processed_lex = None

        # TODO add markers which tokens were split by tokenizer

        # cut/pad the sequences & the bert markers
        for i in range(len(all_ids)):

            # cut until max length, minus 2 to make room for bert markers
            ids = all_ids[i][: self.max_len - 2]
            target_tag = all_target_tags[i][: self.max_len - 2]
            sent_token_ids = token_ids[i][: self.max_len - 2]
            # bert markers and paddings for the rest
            ids = (
                [self.tokenizer.vocab["[CLS]"]] + ids + [self.tokenizer.vocab["[SEP]"]]
            )
            target_tag = self.pad_start_end(self.tag2idx, target_tag)
            sent_token_ids = self.pad_start_end(self.tag2idx, sent_token_ids)

            # pad shortened sequences
            padding_len = self.max_len - len(ids)
            all_ids[i] = ids + ([self.tokenizer.vocab["[PAD]"]] * padding_len)
            all_target_tags[i] = target_tag + ([self.tag2idx["O"]] * padding_len)
            token_ids[i] = sent_token_ids + ([self.tag2idx["O"]] * padding_len)

            # same 3 steps above, but for lexicals only
            if self.lexicals:
                lex_features = {
                    lex: all_lexicals[i][lex][: self.max_len - 2]
                    for lex in all_lexicals[i].keys()
                }
                lex_features = {
                    lex: self.pad_start_end(self.feature_labels[lex], lex_features[lex])
                    for lex in all_lexicals[i].keys()
                }
                processed_lex.append(
                    [
                        lex_features[lex]
                        + ([self.feature_labels[lex]["PAD"]] * padding_len)
                        for lex in all_lexicals[i].keys()
                    ]
                )

        # creating the attention mask
        attention_mask = [[float(i != 0.0) for i in ii] for ii in all_ids]

        return {
            "input_ids": all_ids,
            "attention_mask": attention_mask,
            "labels": all_target_tags,
            "lexicals": processed_lex,
            "token_ids": token_ids,
        }


# if __name__ == "__main__":
#     prep = BertPrep(
#         "data/SEM-2012-SharedTask-CD-SCO-dev-simple-v2-features.csv",
#         ["POS", "Possible_Prefix", "Possible_Suffix"]
#     )
#     processed = prep.preprocess_dataset()
#     pass
