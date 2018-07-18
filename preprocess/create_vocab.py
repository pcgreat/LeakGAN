import pdb
from collections import Counter

from six.moves import cPickle

ltoks = []
with open("composition.txt", "r") as f:
    for line in f:
        toks = line.split("</d>")[0].strip().split()
        ltoks.extend(toks)
vocab_size = 10000

# Generate most common vocab
START_TOKEN = 0
PAD_TOKEN = 1
word = [START_TOKEN, PAD_TOKEN] + [w for w, c in Counter(ltoks).most_common(vocab_size)]
vocab = {}
for i, w in enumerate(word):
    vocab[w] = i

with open("data/vocab_essay.pkl", "wb") as pkl:
    cPickle.dump((word, vocab), pkl)

n = 0
max_seq_length = 100
with open("data/realtrain_essay.txt", "w") as g:
    with open("composition.txt", "r") as f:
        for line in f:
            toks = line.split("</d>")[0].strip().split()
            if any(tok not in vocab for tok in toks):
                continue
            else:
                dtoks = [str(vocab[tok]) for tok in toks]
                dtoks = dtoks[:min(max_seq_length, len(dtoks))] + [str(PAD_TOKEN)] * max(max_seq_length - len(dtoks), 0)
                assert len(dtoks) == max_seq_length, pdb.set_trace()
                g.write(" ".join(dtoks) + "\n")
                n += 1
                if n % 1000 == 0:
                    print(n)
