# Can't use relative import if we want to run this file from the CLI...
from pylt3.ml.rnn.regression import RegressionTrainer
from pylt3.utils.file_helpers import verify_paths

import argparse
from collections import defaultdict
from importlib import import_module
import json
import logging
import numpy as np
from pathlib import Path
import torch

# Make results reproducible
torch.manual_seed(3)
torch.backends.cudnn.deterministic = True
np.random.seed(3)

"""
TODO
 * find solution for OOV word2vec
 * implement optimizer etc through json

"""
def main(opts):
    fs = structure_trainer_files(opts)

    if opts['w2v']['use']:
        load_w2v(opts)

    model = load_model(opts)
    logging.info('Model loaded!')
    logging.info(model)
    trainer = RegressionTrainer(ms=opts['ms'],
                                w2v=opts['w2v'],
                                elmo=opts['elmo'],
                                bert=opts['bert'],
                                model=model,
                                **fs,
                                batch_size=(64, 64, 64))


def load_w2v(opts):
    import gensim
    w2v_p = Path(opts['w2v']['path']).resolve()
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(str(w2v_p), binary=opts['w2v']['binary'])

    opts['w2v']['vocab'] = w2v_model.vocab
    opts['w2v']['weights'] = torch.FloatTensor(w2v_model.vectors)



def load_model(opts):
    """
    Loads an NN model based on the configuration file.

    :param opts: options dict
    :return: initialised NN model
    """
    opts_model = opts['model']
    model_module = import_module(f"pylt3.ml.rnn.regression.models.{opts_model['class']}")
    cls = getattr(model_module, opts_model['class'])
    del opts_model['class']
    model = cls(opts['ms'], opts['w2v'], opts['elmo'], opts['bert'],
                final_drop=opts_model['final_drop'],
                relu=opts_model['relu'])

    return model


def structure_trainer_files(opts):
    # Restructure to {'train': [], 'valid': [], 'test': []}}
    fs = defaultdict(list)
    for input_t, d in opts['trainer']['files'].items():
        # Don't include files if the features are disabled
        if input_t == 'ms' and not opts['ms']['use']:
            continue

        if input_t == 'sentences'\
                and not any([opts['w2v']['use'], opts['elmo']['use'], opts['bert']['use']]):
            continue

        for part, filename in d.items():
            file = Path(filename).resolve()
            if not file.is_file():
                raise ValueError(f"Input file {str(file)} does not exist.")

            fs[f"{part}_files"].append(str(file))

    return fs


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Generate word embeddings from text file.')
    arg_parser.add_argument('config_f', help='Path to JSON file with configuration options.')

    cli_args = arg_parser.parse_args()
    config_f = verify_paths(cli_args.config_f)
    with open(config_f, 'r') as config_fh:
        options = json.load(config_fh)

    main(options)
