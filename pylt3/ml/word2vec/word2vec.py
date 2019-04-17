import argparse
from collections import Counter
import configparser
import logging
from pathlib import Path
from typing import AnyStr, Optional, Union

from gensim import models as gs_models


logging.basicConfig(datefmt='%d-%b %H:%M:%S',
                    format='%(asctime)s - [%(levelname)s]: %(message)s',
                    level=logging.INFO)


class SentenceIterator:
    """ During training, gensim requires an iterator that can't be a generator.
        The reason is because gensim allows multiple training loops, and generators are exhausted
        after one iteration.
        Instead, we build a memory-friendly sentence iterator that does not need to read
        all sentences in memory at once.

        Here we also replace low-frequency words by a placeholder. """
    def __init__(self, input_p, vocab, placeholder):
        self.input_p = input_p
        self.vocab = vocab
        self.placeholder = placeholder

    def __iter__(self):
        with open(str(self.input_p), encoding='utf-8') as fhin:
            for line in fhin:
                tokens = []
                # Get sentence words. Remove empty '' items.
                words = [w for w in line.strip().split() if w != '']

                if not words:
                    continue

                # self.vocab already contains {placeholder} for frequencies lower than a min_count.
                # This means that if a word cannot be found in the vocab, that it is part of the unknown.
                for word in words:
                    if word in self.vocab:
                        tokens.append(word)
                    else:
                        tokens.append(self.placeholder)

                # Return a generator with as output a list of tokens, once per sentence
                yield tokens


def _get_config_args(config_p):
    """ Get parameters from a config file. If it does not exist, use default values.
        These defaults will propagate to the final arguments unless they are overridden by CLI arguments. """
    config_p = Path(config_p).resolve()

    if config_p.exists():
        cfg_parser = configparser.ConfigParser()
        cfg_parser.read(str(config_p))

        # If a parameter is not present in the cfg file, fall back to default values.
        cfg_args = {
            'input_file': cfg_parser.get('io', 'input_file', fallback=''),
            'output_file': cfg_parser.get('io', 'output_file', fallback=''),
            'dims': cfg_parser.getint('word2vec', 'dims', fallback=300),
            'epochs': cfg_parser.getint('word2vec', 'epochs', fallback=5),
            'min_count': cfg_parser.getint('word2vec', 'min_count', fallback=5),
            'window': cfg_parser.getint('word2vec', 'window', fallback=5),
            'replace_unk': cfg_parser.get('other', 'replace_unk', fallback=None),
            'workers': cfg_parser.getint('other', 'workers', fallback=3)
        }
    else:
        # Defaults
        cfg_args = {
            'input_file': '',
            'output_file': '',
            'dims': 300,
            'epochs': 5,
            'min_count': 5,
            'window': 5,
            'replace_unk': None,
            'workers': 3
        }

    return cfg_args


def _get_vocab_counts(input_p, min_count, placeholder):
    """ Count the occurences of every word but add low-frequency words to their own key {placeholder}.
        We do NOT wait to do this when we create the modified sentences:
        1. We want to retrain the generator-type of our sentence iterator, i.e. only yield once every sentence
        2. gensim needs the vocabulary (low-freq removed, placeholder added) at training START so it must be ready. """
    vocab = Counter()
    with open(str(input_p), encoding='utf-8') as fhin:
        sentence_nr = 0
        for line in fhin:
            sentence_nr += 1

            if sentence_nr % 10000 == 0:
                logging.info(f"PROGRESS: generating vocab frequency at sentence #{sentence_nr}")

            # Get sentence words. Remove empty '' items.
            words = [w for w in line.strip().split() if w != '']

            if not words:
                continue

            for word in words:
                vocab[word] += 1

    # Do the low-frequency clean-up
    logging.info('PROGRESS: finding low-frequency words...')
    unk_freqs = [(words, counts) for words, counts in vocab.items() if counts < min_count]
    unk_words, unk_counts = zip(*unk_freqs)

    # remove unknown words
    for unk_word in unk_words:
        vocab.pop(unk_word, None)

    # add {placeholder} with unk counts
    vocab[placeholder] = sum(unk_counts)

    logging.info(f"DONE: generating vocab frequency. {vocab[placeholder]} unknown words.")
    return vocab


def _merge_args(cli_args, cfg_args):
    """ Command-line arguments overwrite configuration file parameters. """
    # Remove None values from CLI arguments
    cli_args = {k: v for k, v in cli_args.items() if v is not None}
    args = {**cfg_args, **cli_args}

    args['input_file'], args['output_file'] = _verify_paths(args['input_file'], args['output_file'])

    return args


def test_word2vec(model_f: Union[Path, AnyStr]):
    logging.info("TESTING MODEL. If unexpected values, corpus might be from a different domain or too small.")

    model = gs_models.KeyedVectors.load_word2vec_format(str(model_f))
    try:
        logging.info(f"Most similar: positive ['king', 'woman'], negative ['man']:"
                     f" {model.wv.most_similar(positive=['king', 'woman'], negative=['man'])}")
        logging.info(f"Similarity woman-man:"
                     f" {model.wv.similarity('man', 'woman')}")
    except KeyError:
        raise ValueError('Test failed because of unkown words. Your corpus might be from a different domain'
                         ' or very small.')


def train_word2vec(input_file: Union[Path, AnyStr], output_file: Union[Path, AnyStr],
                   dims: int = 100, epochs: int = 5, min_count: int = 5, window: int = 5,
                   replace_unk: Optional[str] = None, workers: int = 3, **_) -> Path:
    """ Train word2vec model and save its vectors to an output file.

        When replace_unk the behaviour is different than normal. We want to replace low-frequency words
        with a placeholder and feed the resulting sentences to gensim.
        We don't want to generate the frequencies twice (once manually, and once automatically by gensim) so we
        build the vocabulary from the frequencies that we already had. This saves some time. """

    input_file, output_file = _verify_paths(input_file, output_file)

    if replace_unk:
        # vocab is a str->int dict with frequency counts. Frequencies lower than min_count are summed in
        # the {placeholder} key
        vocab = _get_vocab_counts(input_file, min_count, placeholder=replace_unk)
        sentences = SentenceIterator(input_file, vocab, replace_unk)
        model = gs_models.Word2Vec(size=dims,
                                   window=window,
                                   min_count=min_count,
                                   iter=epochs,
                                   workers=workers)

        model.build_vocab_from_freq(vocab)
        model.train(sentences,
                    total_examples=model.corpus_count,
                    total_words=sum(vocab.values()),
                    epochs=model.epochs)

    else:
        sentences = gs_models.word2vec.LineSentence(str(input_file))

        model = gs_models.Word2Vec(sentences=sentences,
                                   size=dims,
                                   window=window,
                                   min_count=min_count,
                                   iter=epochs,
                                   workers=workers)

    # save only the word vectors in text format
    model.wv.save_word2vec_format(str(output_file), binary=False)
    del model

    # basic test for the model
    test_word2vec(output_file)

    return Path(output_file)


def _verify_paths(input_f, output_f):
    """ Ensure that input and output file arguments are set. """
    if not input_f:
        raise ValueError(f"'input_file' cannot be undefined.")
    else:
        input_f = Path(input_f).resolve()

    if not output_f:
        raise ValueError(f"'output_file' cannot be undefined.")
    else:
        output_f = Path(output_f).resolve()
        output_f.parent.mkdir(exist_ok=True, parents=True)

    return input_f, output_f


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Generate word embeddings from text file.')

    arg_parser_io = arg_parser.add_argument_group('IO', description='I/O settings.')
    arg_parser_io.add_argument('--ini', default='config.ini', help="Path to a configuration file. Command line"
                                                                   " arguments have precedence over config file.")
    arg_parser_io.add_argument('-i', '--input_file', help="Path to input file.")
    arg_parser_io.add_argument('-o', '--output_file', help="Path to output file.")

    arg_parser_w2v = arg_parser.add_argument_group('word2vec', description='Word2vec processing parameters.')
    arg_parser_w2v.add_argument('-c', '--min_count', type=int,
                                help="Threshold for token frequency. If a token occurs less than 'min_count',"
                                     " it will not be included. Also see 'replace_unk' under 'other'.")
    arg_parser_w2v.add_argument('-d', '--dims', type=int,
                                help="Number of dimensions for the word2vec representation.")
    arg_parser_w2v.add_argument('-e', '--epochs', type=int, help="Number of training epochs.")
    arg_parser_w2v.add_argument('-w', '--window', type=int,
                                help="Maximum distance between the current and predicted word within a sentence.")

    arg_parser_other = arg_parser.add_argument_group('other')
    # Set default 'replace_unk' to None so that a standard False value does not overwrite CFG.
    arg_parser_other.add_argument('-u', '--replace_unk',
                                  help="Rather than removing words with a frequency lower than 'min_count', replace"
                                       " these words by a given token.")
    arg_parser_other.add_argument('-n', '--workers', type=int, help="Number of threads to use.")

    # Get args from config file and command line, and merge.
    commandline_args = vars(arg_parser.parse_args())
    config_args = _get_config_args(commandline_args['ini'])
    merged_args = _merge_args(commandline_args, config_args)

    # Train word2vec.
    train_word2vec(**merged_args)
