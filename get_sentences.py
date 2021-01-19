from feature_embeddings import BertPrep, SentenceGetter

train_prep = BertPrep("data/SEM-2012-SharedTask-CD-SCO-training-simple.txt")
getter = SentenceGetter(train_prep.data)
sentences = [[s[0] for s in sent] for sent in getter.sentences]

for i, s in enumerate(sentences):
    print(" ".join(s))
