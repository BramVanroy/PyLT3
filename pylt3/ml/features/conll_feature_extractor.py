import numpy as np
from pathlib import Path
import pickle
from typing import AnyStr, Optional, Union
from collections import defaultdict


def _line_props(line):
    token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = line.split('\t')

    feats = feats.split('|')

    return upos, xpos, feats, deps


def _merge_dict2set(di, dj):
    for k, v in dj.items():
        di[k].add(v)

    return di


def get_feats_ids(*pfins: Union[Path, AnyStr], save: Optional[Union[Path, AnyStr]]=None, encoding: str='utf-8'):
    upos_all = set()
    xpos_all = set()
    feats_all = set()
    deps_all = set()

    props2idx = {'upos': defaultdict(), 'xpos': defaultdict(), 'deps': defaultdict(), 'feats': defaultdict()}
    prop_idx = 0

    for pfin in pfins:
        with open(str(pfin), 'r', encoding=encoding) as fhin:
            for line in fhin:
                line = line.strip()

                if line == '':
                    continue

                upos, xpos, feats, deps = _line_props(line)

                if upos not in upos_all:
                    props2idx['upos'][upos] = prop_idx
                    prop_idx += 1
                    upos_all.add(upos)

                if xpos not in xpos_all:
                    props2idx['xpos'][xpos] = prop_idx
                    prop_idx += 1
                    xpos_all.add(xpos)

                if deps not in deps_all:
                    props2idx['deps'][deps] = prop_idx
                    prop_idx += 1
                    deps_all.add(deps)

                for val in feats:
                    if val not in feats_all:
                        feats_all.add(val)
                        props2idx['feats'][val] = prop_idx
                        prop_idx += 1

    # Convert to regular dict for smaller footprint.
    for k in props2idx.keys():
        props2idx[k] = dict(props2idx[k])

    props2idx = dict(props2idx)

    if save is not None:
        with open(str(save), 'wb') as fhout:
            pickle.dump(props2idx, fhout)

    print(props2idx)
    return props2idx


# convert to numpy
def file2feats(pfin: Union[Path, AnyStr], pffeats: Union[Path, AnyStr], pfout: Union[Path, AnyStr],
               encoding: str='utf-8'):

    with open(str(pffeats), 'rb') as fh_feats:
        ms_feats = pickle.load(fh_feats)

    nro_feats = sum([len(v) for v in ms_feats.values()])
    nro_sentences = 0
    with open(str(pfin), encoding=encoding) as fhin, open(str(pfout), 'w', encoding='utf-8') as fhout:
        sentence = []
        nro_tokens = 0
        next_sentence = False
        for line in fhin:
            line = line.strip()

            # When skipping (next_sentence), stop skipping when reaching empty line
            if line == '':
                if sentence:
                    nro_sentences += 1
                    print(f"Processing line {nro_sentences}...\r")

                    sentence = [list(map(str, token)) for token in sentence]

                    s = ''
                    for token in sentence:
                        s += ' '.join(token)
                        s += '\t'

                    fhout.write(s.strip() + '\n')

                    sentence = []
                    nro_tokens = 0
                    next_sentence = False
                continue
            elif next_sentence:
                continue

            token = np.zeros(nro_feats, dtype=np.int8)
            nro_tokens += 1
            upos, xpos, feats, deps = _line_props(line)

            token[ms_feats['upos'][upos]] = 1
            token[ms_feats['xpos'][xpos]] = 1
            token[ms_feats['deps'][deps]] = 1

            for val in feats:
                token[ms_feats['feats'][val]] = 1

            sentence.append(token)


if __name__ == '__main__':
    file2feats(r'C:\wsl-shared\cross-conll\test\conll.txt',
               r'C:\Python\projects\PyLT3\data\dpc\ml\other\dpc.feats2idx.pickle',
               r'C:\wsl-shared\cross-conll\test\conll-feats.test')
