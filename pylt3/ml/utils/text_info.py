from collections import Counter
import logging
import math
import numpy as np
from pathlib import Path
from statistics import mean, median
import typing as t
from typing import Optional, Union, Dict, AnyStr, Tuple


logging.basicConfig(
    format='[%(levelname)s] %(asctime)s: %(message)s',
    level=logging.INFO)


# Sentence Length
# ---------------
def sentence_length_info(pfin: Optional[Union[Path, AnyStr]] = None, pdin: Optional[Union[Path, AnyStr]] = None,
                         recursive: bool = False, encoding: str = 'utf-8', ext: str = '', verbose: int = 2):
    sentence_lengths = []
    extremes_counter = Counter()
    if pfin:
        p_str = str(pfin)
        s_lengths, extreme_c = _f_sentence_length(Path(pfin).resolve(), encoding=encoding, verbose=verbose)
        sentence_lengths.extend(s_lengths)
        extremes_counter += extreme_c
    elif pdin:
        p_str = str(pdin)
        # Use recursion or not
        sub_path = f"**/*{ext}" if recursive else "*{ext}"
        for path_fin in Path(pdin).resolve().glob(sub_path):
            s_lengths, extreme_c = _f_sentence_length(path_fin, encoding=encoding, verbose=verbose)
            sentence_lengths.extend(s_lengths)
            extremes_counter += extreme_c
    else:
        raise ValueError("One of 'pfin' or 'pdin' arguments must be provided.")

    dist_info = _dist_info(sentence_lengths)

    if verbose:
        logging.info(f"Sentence length distribution for {p_str}:")
        logging.info(f"\t- max: {dist_info['max']}")
        logging.info(f"\t- min: {dist_info['min']}")
        logging.info(f"\t- mean: {dist_info['mean']}")
        logging.info(f"\t- median: {dist_info['median']}")
        logging.info(f"\t- std: {dist_info['std']}")
        logging.info(f"\t- tokens<3: {extremes_counter[3]}")
        logging.info(f"\t- 100<tokens<1000: {extremes_counter[100]}")
        logging.info(f"\t- 1000<tokens: {extremes_counter[1000]}")

    return dist_info


def _dist_info(distribution: t.List[int]) -> Dict[str, float]:

    dist_np = np.asarray(distribution)

    info = {
        'max': np.max(dist_np),
        'min': np.min(dist_np),
        'mean': np.mean(dist_np),
        'median': np.median(dist_np),
        'std': np.std(dist_np)
    }

    return info


def _f_sentence_length(pfin: Union[Path, AnyStr], encoding: str, verbose: int) -> Tuple[t.List[int], t.Counter[int]]:
    sentence_lengths = []
    extremes_counter = Counter()
    # We don't know the size of the file, so rather than loading it in memory, be lazy
    with open(pfin, 'r', encoding=encoding) as fhin:
        line_nr = 0
        for line in fhin:
            line = line.strip()
            if line == '':
                continue

            line_nr += 1
            if verbose and line_nr % 10000 == 0:
                logging.info(f"PROCESSING: line {line_nr}")
            s_length = len(line.split())
            sentence_lengths.append(s_length)

            # Count long sentence, i.e. more than 100 or even 1000 tokens
            if s_length > 1000:
                if verbose > 1:
                    logging.info(f"Extremely long (>1000) sentence detected ({s_length}): line {line_nr}:\n{line}")
                extremes_counter[1000] += 1
            elif s_length > 100:
                if verbose > 1:
                    logging.info(f"Long sentence (100<<100) detected ({s_length}): line {line_nr}:\n{line}")
                extremes_counter[100] += 1
            elif s_length < 3:
                if verbose > 1:
                    logging.info(f"Short sentence (<3) detected ({s_length}): line {line_nr}:\n{line}")
                extremes_counter[3] += 1

    return sentence_lengths, extremes_counter


if __name__ == '__main__':
    sentence_length_info(pfin="C:\Python\projects\pylt3-ml\pylt3-ml\word2vec\data\input\dpc+news2017.norm.tok.low.en")
