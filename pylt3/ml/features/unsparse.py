from pathlib import Path
import sys

""" 'unsparse' data, i.e. converting a line of indices to one-hot encoded vectors.
    Example input:
        s = '[25 90 146]	[5 69 113]	[20 94 134]	[40 64 134]	[31 69 113]'
    """


def unsparse(line):
    tokens = line.split('\t')
    # Remove brackets
    tokens = [t[1:-1] for t in tokens]
    tokens = [list(map(int, t.split())) for t in tokens if t != '-1']

    return tokens


def unsparse_file(fin, dim, fhout=sys.stdout):
    pfin = Path(fin).resolve()
    if fhout != sys.stdout:
        pfout = Path(fhout).resolve()
        fhout = open(pfout, 'w', encoding='utf-8')

    with open(pfin, encoding='utf-8') as fhin:
        for line in fhin:
            line = line.strip()
            tokens = unsparse(line)
            s = unsparse_to_string(tokens, dim)
            fhout.write(s + '\n')

    if fhout != sys.stdout:
        fhout.close()


def unsparse_to_string(tokens, dim):
    s = ''
    for idx, t in enumerate(tokens):
        if idx > 0:
            s += '\t'
        n = ['0'] * dim
        for i in t:
            n[i] = '1'
        s += ' '.join(n)

    return s


if __name__ == '__main__':
    unsparse_file(r'C:\wsl-shared\arda-dataset\MS_features\train_sparse_features.txt',
                  147,
                  r'C:\wsl-shared\arda-dataset\MS_features_onehot\train_unsparse_features.txt')