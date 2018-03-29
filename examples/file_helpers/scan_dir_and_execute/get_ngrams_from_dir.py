"""
Given a directory, a filename to write to, and an integer, get a dictionary with all n-grams up to the given integer.
For instance, if you want unigrams, bigrams, and trigrams then the integer should be `3`. The resulting ngrams are
written to the given out file with their respective frequencies.
"""
from pylt3 import file_helpers

from nltk import word_tokenize
from nltk.util import ngrams

from collections import Counter
from pathlib import Path
import sys


def create_ngrams(file, i, freq_dict):
    for m in range(1, i+1):
        if m not in gram_freqs:
            freq_dict[m] = Counter([])

        text = open(file).read()
        tokenised = word_tokenize(text)
        grams = ngrams(tokenised, m)
        freq_dict[m] += Counter(grams)


if __name__ == "__main__":
    my_dir = str(Path(sys.argv[1]))
    fout = str(Path(sys.argv[2]))
    width = int(sys.argv[3])

    gram_freqs = {}
    file_helpers.scan_dir_and_execute(my_dir, lambda file: create_ngrams(file, width, gram_freqs), verbose=2)

    with open(fout, 'w') as f:
        for n in gram_freqs:
            for ngram in gram_freqs[n]:
                f.write(" ".join(map(str, ngram)) + "\t" + str(gram_freqs[n][ngram]) + "\n")
