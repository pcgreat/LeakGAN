from six.moves import cPickle
import pdb


with open("../save/vocab_cotra.pkl", "rb") as pkl:
    word, vocab = cPickle.load(pkl)
    pdb.set_trace()