import pdb
from collections import Counter

from six.moves import cPickle

ltoks = []
with open("../data/zuowen_sents.txt", "r") as f:
    for line in f:
        toks = list(line.strip().replace(" ",""))
        ltoks.extend(toks)
vocab_size = 5000

# Generate most common vocab
PAD_TOKEN = "<PAD>"
# START_TOKEN = 1
END_TOKEN= "<EOS>"
word = [PAD_TOKEN, END_TOKEN] + [w for w, c in Counter(ltoks).most_common(vocab_size)]

vocab = {}
for i, w in enumerate(word):
    vocab[w] = i

with open("../data/vocab_essay.pkl", "wb") as pkl:
    cPickle.dump((word, vocab), pkl)

n = 0
max_seq_length = 30
with open("../data/realtrain_essay.txt", "w") as g:
    with open("../data/zuowen_sents.txt", "r") as f:
        for line in f:
            toks = list(line.split("</d>")[0].strip().replace(" ",""))
            if any(tok not in vocab for tok in toks):
                continue
            else:
                if len(toks) > max_seq_length or len(toks) < 10:
                    continue
                dtoks = [str(vocab[tok]) for tok in toks]
                dtoks = dtoks[:(max_seq_length-1)] + [str(vocab[END_TOKEN])]
                dtoks = dtoks + [str(vocab[PAD_TOKEN])] * (max_seq_length - len(dtoks))
                assert len(dtoks) == max_seq_length, pdb.set_trace()
                g.write(" ".join(dtoks) + "\n")
                n += 1
                if n % 1000 == 0:
                    print(n)
