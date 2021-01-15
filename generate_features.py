import pandas as pd
import spacy
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer

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
    df['Token_lower']=df.apply(lambda row: row['Token'].lower(), axis=1)
    #Adds columns for the lemma, pos-tag, snowballstem and porterstem
    df['Lemma'], df['POS'], df['SnowballStemmer'], df['PorterStemmer'] = zip(*df.apply(lambda row: linguistic_features(row['Token_lower']), axis=1))
    #Adds columns with a binary if the word contains a possible negation prefix or suffix
    df['Possible_Prefix'] = df.apply(lambda row: possible_negation_prefix(row['Token_lower']), axis=1)
    df['Possible_Suffix'] = df.apply(lambda row: possible_negation_suffix(row['Token_lower']), axis=1)
    #Adds new columns for the previous and next lemma and pos-tag
    df['Prev_Lemma'] = df['Lemma'].shift(periods=-1)
    df['Next_Lemma'] = df['Lemma'].shift(periods=1)
    df['Prev_POS'] = df['POS'].shift(periods=-1)
    df['Next_POS'] = df['POS'].shift(periods=1)
    return df

if __name__ == '__main__':
    #Load spacy model
    nlp = spacy.load("en_core_web_sm")
    #Load data csv
    df=pd.read_csv('SEM-2012-SharedTask-CD-SCO-simple/SEM-2012-SharedTask-CD-SCO-dev-simple.txt', delimiter='\t', names=['Chapter', 'Sentence Number', 'Word Number', 'Token', 'Annotation'])
    df=generate_features(df)
    #Store generated conll file
    df.to_csv('SEM-2012-SharedTask-CD-SCO-simple/conll_extended.csv', sep='\t')
    print(df.head())