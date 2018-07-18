import pdb
from collections import Counter
from random import shuffle

from six.moves import cPickle

ltoks = []
with open("../data/zuowen_sents.txt", "r") as f:
    for line in f:
        toks = list(line.strip().replace(" ", ""))
        ltoks.extend(toks)
vocab_size = 5000

# Generate most common vocab
PAD_TOKEN = "<PAD>"
# START_TOKEN = 1
END_TOKEN = "<EOS>"
word = [PAD_TOKEN, END_TOKEN] + [w for w, c in Counter(ltoks).most_common(vocab_size)]

vocab = {}
for i, w in enumerate(word):
    vocab[w] = i

with open("../data/vocab_essay.pkl", "wb") as pkl:
    cPickle.dump((word, vocab), pkl)


max_seq_length = 30
lines = open("../data/zuowen_sents.txt", "r").readlines()
train_cnt = int(len(lines) * 0.8)
shuffle(lines)


def write_line(g, line):
    global n
    toks = list(line.split("</d>")[0].strip().replace(" ", ""))
    if any(tok not in vocab for tok in toks):
        return
    else:
        if len(toks) > max_seq_length or len(toks) < 10:
            return
        dtoks = [str(vocab[tok]) for tok in toks]
        dtoks = dtoks[:(max_seq_length - 1)] + [str(vocab[END_TOKEN])]
        dtoks = dtoks + [str(vocab[PAD_TOKEN])] * (max_seq_length - len(dtoks))
        assert len(dtoks) == max_seq_length, pdb.set_trace()
        g.write(" ".join(dtoks) + "\n")
        n += 1
        if n % 1000 == 0:
            print(n)


with open("../data/realtrain_essay_train.txt", "w") as g:
    n = 0
    for line in lines[:train_cnt]:
        write_line(g, line)

with open("../data/realtrain_essay_eval.txt", "w") as g:
    n = 0
    for line in lines[train_cnt:]:
        write_line(g, line)

