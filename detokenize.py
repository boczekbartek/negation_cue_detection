# %%
import subprocess
import os

from transformers.models.transfo_xl.tokenization_transfo_xl import tokenize_numbers


def setup_stanford_nlp():
    """ Setup StanfordNLP java Tool. """
    here = os.path.abspath(os.path.dirname("__file__"))
    os.environ[
        "CLASSPATH"
    ] = f"{here}/tools/stanford-parser-full-2020-11-17/stanford-parser.jar"


def detokenize(sentence: str):
    setup_stanford_nlp()
    cmd = "java edu.stanford.nlp.process.PTBTokenizer -untok".split()
    stanford_nlp_parser = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if sentence.endswith("\n"):
        ends_with_newline = True
    else:
        ends_with_newline = False
        sentence = sentence + "\n"

    detokenized_sentence, _ = stanford_nlp_parser.communicate(sentence)
    if not ends_with_newline:
        detokenized_sentence = detokenized_sentence[:-1]
    return detokenized_sentence


# %%
from feature_embeddings import BertPrep, SentenceGetter

train_prep = BertPrep("data/SEM-2012-SharedTask-CD-SCO-training-simple.txt")
getter = SentenceGetter(train_prep.data)
sentences = [[s[0] for s in sent] for sent in getter.sentences]
labels = [[s[1] for s in sent] for sent in getter.sentences]
from transformers import BertTokenizer
tok = BertTokenizer.from_pretrained('bert-base-uncased')

# %%

m = 20
for i, (s, l) in enumerate(zip(sentences[0:m], labels[0:m])):
    
    pairs_ori = [(t,ll) for t,ll in zip(s,l)]
    # print("Original:")
    # print('S:',"  ".join(s))
    # print('L:', " ".join(l))
    # tokens = tok.tokenize(' '.join(s))
    # # print('Tok:',' '.join(tokens))
    # print(len(tokens))
    # print("------- Detok:")
    detok_s = detokenize(" ".join(s))
    detok_tok = detok_s.split(' ')
    pairs = [(t,ll) for t,ll in zip(detok_tok,l)]
    if len(pairs) != len(pairs_ori) and 'B-NEG' in l:
        print(i,'| ori = ', len(pairs),'detok =', len(pairs_ori))
        for l1,l2 in zip(pairs_ori, pairs):
            print(l1, l2)
        print(l)
        print(s)
        print(detok_tok)
    # print('Detok S:',detok_s) 
    # detok_tokens = tok.tokenize(detok_s)
    # print(len(detok_tokens))
    # print('Detok tokens:', ' '.join(detok_tokens))
    print("----------------------------------")

# %%

?zip
# %%

# %%
