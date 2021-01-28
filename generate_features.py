import logging
import argparse

import pandas as pd
import spacy
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer


def possible_negation_prefix(text):
    """
    Checks if the texts contains a possible negation prefix
    :param text: string containing a token

    :rtype: bool
    :return: True if the texts starts with a possible negation prefix, False if not
    """
    prefixes = ("de", "dis", "im", "in", "ir", "il", "non", "un", "mis")
    # Length is mentioned to limit wrong prefix recongnition such as "none" or "mist"
    return text.startswith(prefixes) and len(text) >= 5


def possible_negation_suffix(text):
    """
    Checks if the texts contains a possible negation suffix

    :param text: string containing a token

    :rtype: bool
    :return: True if the texts ends with a possible negation suffix, False if not
    """
    suffixes = ("less",)
    # length is mentioned so it doesn't consider "less" as containing the suffix
    return text.endswith(suffixes) and len(text) >= 5


def generate_features(df, spacy_model, language):
    """
    Extends the dataframe by adding columns for newly generated features.
    Lemma, pos-tag, snowballstem, porterstem, if it contains a possible negation prefix or suffix, next and previous lemma, next and previous pos-tag

    :param dataframe df: dataframe that contains the presented data in conll-format
    :param spacy_model str: name of SpaCy model used for features extractiom
    :param language str: language used as parameter of Snowball Stemmer

    :rtype: dataframe
    :return: dataframe that contains 11 more columns containing the afformentioned features
    """
    logging.info("Loading Spacy model...")
    nlp = spacy.load(spacy_model)

    # Makes all tokens lowercase
    logging.info("Lowercase")
    df["token_lower"] = df["token"].str.lower()

    logging.info("Lemma, pos")
    spacy_pipe = nlp.pipe(df["token_lower"].values, disable=["ner", "parser"])
    features_gen = ((doc[0].lemma_, doc[0].pos_) for doc in spacy_pipe)
    df["lemma"], df["pos"] = zip(*features_gen)

    # Prepare stemmers
    logging.info("Loading Snowball Stemmer...")
    snow = SnowballStemmer(language=language)

    logging.info("Snowball stemmer")
    df["snowballStemmer"] = df.apply(lambda row: snow.stem(row["token_lower"]), axis=1)

    logging.info("Loading Porter Stemmer...")
    port = PorterStemmer()

    logging.info("Porter stemmer")
    df["porterStemmer"] = df.apply(lambda row: port.stem(row["token_lower"]), axis=1)

    # Adds columns with a binary if the word contains a possible negation prefix or suffix
    logging.info("Prefix")
    df["possible_prefix"] = df.apply(
        lambda row: possible_negation_prefix(row["token_lower"]), axis=1
    )

    logging.info("Suffix")
    df["possible_suffix"] = df.apply(
        lambda row: possible_negation_suffix(row["token_lower"]), axis=1
    )

    # Adds new columns for the previous and next lemma and pos-tag
    logging.info("Add prev/next shifts")
    df["prev_Lemma"] = df["lemma"].shift(periods=1)
    df["next_Lemma"] = df["lemma"].shift(periods=-1)
    df["prev_pos"] = df["pos"].shift(periods=1)
    df["next_pos"] = df["pos"].shift(periods=-1)
    return df


def run_generate_features(
    data_file, features_file, spacy_model="en_core_web_sm", language="english"
):
    # Load spacy model
    logging.info(f"Loading data: {data_file}")
    df = pd.read_csv(
        data_file, delimiter="\t", names=["corpus", "n_sent", "n_word", "token", "tag"]
    )

    logging.info("Generating features...")
    df = generate_features(df, spacy_model=spacy_model, language=language)

    # Store generated conll file
    logging.info(f"Saving features to: {features_file}")
    df.to_csv(features_file, sep="\t")
    logging.info("\n" + str(df.head()))


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate lexical features")
    parser.add_argument("data_file", type=str, help="Data in CoNLL format.")
    parser.add_argument("features_file", type=str, help="Path out output file")
    parser.add_argument(
        "--spacy-model",
        required=False,
        type=str,
        default="en_core_web_sm",
        help="Name of SpaCy model",
    )
    parser.add_argument(
        "--language",
        required=False,
        type=str,
        default="english",
        help="Language used in PorterStemmer",
    )
    args = parser.parse_args()

    run_generate_features(**vars(args))

