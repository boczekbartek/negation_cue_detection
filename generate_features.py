import pandas as pd
import spacy
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer

nlp = spacy.load("en_core_web_sm")
#Prepare stemmers
snow = SnowballStemmer(language='english')
port = PorterStemmer()


def linguistic_features(text):
    """
    Generates the lemma, pos-tag, snowball-stem and porter-stem of a word

    :param text: string of the token
    :rtype: strings
    :return: multiple strings containing the lemma, pos-tag, snowball-stem and porter-stem
    """
    doc=nlp(text)
    token=doc[0]
    return token.lemma_, token.pos_, snow.stem(text), port.stem(text)

def possible_negation_prefix(text):
    """
    Checks if the texts contains a possible negation prefix
    :param text: string containing a token

    :rtype: bool
    :return: True if the texts starts with a possible negation prefix, False if not
    """
    prefixes=['de','dis', 'im', 'in', 'ir', 'il', 'non', 'un', 'mis']
    #Length is mentioned to limit wrong prefix recongnition such as "none" or "mist"
    if text.startswith(tuple(prefixes)) and len(text) >= 5:
        return True
    else:
        return False

def possible_negation_suffix(text):
    """
    Checks if the texts contains a possible negation suffix

    :param text: string containing a token

    :rtype: bool
    :return: True if the texts ends with a possible negation suffix, False if not
    """
    suffixes=['less']
    #length is mentioned so it doesn't consider "less" as containing the suffix
    if text.endswith(tuple(suffixes)) and len(text) >= 5:
        return True
    else:
        return False

def generate_features(df):
    """
    Extends the dataframe by adding columns for newly generated features.
    Lemma, pos-tag, snowballstem, porterstem, if it contains a possible negation prefix or suffix, next and previous lemma, next and previous pos-tag

    :param dataframe df: dataframe that contains the presented data in conll-format

    :rtype: dataframe
    :return: dataframe that contains 11 more columns containing the afformentioned features
    """
    #Makes all tokens lowercase
    df['token_lower']=df.apply(lambda row: row['token'].lower(), axis=1)
    #Adds columns for the lemma, pos-tag, snowballstem and porterstem
    df['lemma'], df['pos'], df['snowballStemmer'], df['porterStemmer'] = zip(*df.apply(lambda row: linguistic_features(row['token_lower']), axis=1))
    #Adds columns with a binary if the word contains a possible negation prefix or suffix
    df['possible_prefix'] = df.apply(lambda row: possible_negation_prefix(row['token_lower']), axis=1)
    df['possible_suffix'] = df.apply(lambda row: possible_negation_suffix(row['token_lower']), axis=1)
    #Adds new columns for the previous and next lemma and pos-tag
    df['prev_Lemma'] = df['lemma'].shift(periods=1)
    df['next_Lemma'] = df['lemma'].shift(periods=-1)
    df['prev_pos'] = df['pos'].shift(periods=1)
    df['next_pos'] = df['pos'].shift(periods=-1)
    return df

if __name__ == '__main__':
    #Load spacy model
    nlp = spacy.load("en_core_web_sm")
    #Load data csv
    df=pd.read_csv('SEM-2012-SharedTask-CD-SCO-simple/SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt', delimiter='\t', names=["corpus", "n_sent", "n_word", "token", "tag"])
    print(df['Annotation'].value_counts())
    df=generate_features(df)
    #Store generated conll file
    df.to_csv('SEM-2012-SharedTask-CD-SCO-simple/conll_extended.csv', sep='\t')
    print(df.head())