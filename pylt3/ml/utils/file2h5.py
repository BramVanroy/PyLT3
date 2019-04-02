from pathlib import Path
from typing import AnyStr, Optional, Union

import h5py
import numpy as np


def _pad_seq(l, min_size, pad_str=''):
    l += (min_size - len(l)) * [pad_str]

    return l

def file2h5(pfin: Union[Path, AnyStr], pfout: Union[Path, AnyStr], split_on: Optional[str] = None,
            cast_to: Optional=None, min_size: Optional[int]=None, max_size: Optional[int]=None, encoding: str='utf-8'):
    lines = []

    nro_lines = 0
    with open(str(pfin), 'r', encoding=encoding) as fhin:
        for line in fhin:
            line = line.strip()

            if line == '':
                continue

            nro_lines += 1

            if split_on:
                line = line.split(split_on)
                line = list(filter(None, line))
            else:
                line = [line]

            if cast_to is not None:
                line = [cast_to(l) for l in line]

            if min_size is not None and len(line) < min_size:
                line = _pad_seq(line, min_size)
            elif max_size and len(line) > max_size:
                line = line[:max_size]

            print(f"Processed line {nro_lines}...\r")
            lines.append(np.array(line))


    lines = np.array(lines)

    print('Saving file...')
    with h5py.File(str(pfout), 'w') as fhout:
        fhout.create_dataset('utils', data=lines)

    return lines


if __name__ == '__main__':
    in_p = Path(r'C:\Python\projects\PyLT3\data\dpc\prep\train.tok.low.en')
    out_p = Path(r'C:\Python\projects\PyLT3\data\dpc\ml\train\dpc.train.sents.h5')

    data = file2h5(in_p, out_p, split_on=' ', min_size=100, max_size=100)
    print(data)
