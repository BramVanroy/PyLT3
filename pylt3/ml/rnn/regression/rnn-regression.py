import argparse
from importlib import import_module
import json
import logging
import os
from pathlib import Path
import random
from shutil import copy, move

from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim

# Can't use relative import if we want to run this file from the CLI...
from pylt3.ml.rnn.regression import RegressionTrainer
from pylt3.utils.file_helpers import verify_paths


# Make results reproducible
def set_seed():
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(3)
    random.seed(3)
    os.environ['PYTHONHASHSEED'] = str(3)


def main(old_opts, config_p):
    output_p = Path(old_opts['trainer'].pop('output_dir')).resolve()
    for hidden_dim in (256, 384, 512):
        for dropout in (0, 0.1, 0.2, 0.3, 0.4):
            opts = deepcopy(old_opts)

            # opts['model']['relu']['use'] = relu

            opts['model']['final_drop'] = dropout
            opts['ms']['recurrent_layer']['dropout'] = dropout
            opts['w2v']['recurrent_layer']['dropout'] = dropout
            opts['ft']['recurrent_layer']['dropout'] = dropout

            opts['ms']['recurrent_layer']['dim'] = hidden_dim
            opts['w2v']['recurrent_layer']['dim'] = hidden_dim
            opts['ft']['recurrent_layer']['dim'] = hidden_dim

            set_seed()

            if opts['w2v']['use']:
                load_w2v(opts)

            if opts['ft']['use']:
                load_fasttext(opts)

            model = load_model(opts)
            logging.info('Model loaded!')
            logging.info(model)

            criterion = get_criterion(opts['criterion'])
            optimizer, optim_name = get_optim(opts['optimizer'], model)
            scheduler = get_scheduler(opts['scheduler'], optimizer)

            trainer = RegressionTrainer(ms=opts['ms'],
                                        w2v=opts['w2v'],
                                        fasttext=opts['ft'],
                                        elmo=opts['elmo'],
                                        bert=opts['bert'],
                                        model=model,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        **opts['trainer'].pop('files'),
                                        batch_size=(200, 200, 200))

            best_model_f, fig = trainer.train(**opts['trainer'])
            best_model_p = Path(best_model_f).resolve()
            loss, pearson = trainer.test()

            s = get_output_prefix(loss, pearson, optim_name, opts)
            model_out = output_p.joinpath(s+'model.pth')
            config_out = output_p.joinpath(s+'config.json')

            # copy config file to output dir
            copy(config_p, config_out)
            # move output model to output_dir
            move(best_model_p, model_out)
            # Save plot
            fig.savefig(output_p.joinpath(s+'plot.png'))


def get_output_prefix(loss, pearson, optim_name, opts):
    s = f"loss{loss:.2f}-pearson{pearson:.2f}-"
    s += f"{optim_name}-lr{opts['optimizer']['lr']:.0E}-"

    used = []
    for feat in ('ms', 'w2v', 'ft'):
        if opts[feat]['use']:
            feat_s = f"{str(opts[feat]['dim'])}{feat}-{opts[feat]['recurrent_layer']['type']}-"
            feat_s += f"h{str(opts[feat]['recurrent_layer']['dim'])}"
            if 'bidirectional' in opts[feat]['recurrent_layer'] and opts[feat]['recurrent_layer']['bidirectional']:
                feat_s += '-bi'
            used.append(feat_s)

    if opts['elmo']['use']:
        used.append('elmo')
    if opts['bert']['use']:
        used.append('bert')

    s += '+'.join(used) + '+drop' + str(opts['model']['final_drop']) + '-'

    if opts['model']['relu']['use']:
        s += 'relu-'

    return s


def get_criterion(crit_str):
    if crit_str == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError


def get_optim(optim_obj, model):
    optim_name = optim_obj.pop('name')
    if optim_name.lower() == 'bert-adam':
        from pytorch_pretrained_bert.optimization import BertAdam
        return BertAdam([p for p in model.parameters() if p.requires_grad],
                        **optim_obj), optim_name
    elif optim_name.lower() == 'adam':
        return optim.Adam([p for p in model.parameters() if p.requires_grad],
                          **optim_obj), optim_name
    else:
        raise ValueError


def get_scheduler(sched_obj, optimizer):
    if sched_obj.pop('use'):
        sched_name = sched_obj.pop('name')
        if sched_name.lower() == 'reducelronplateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **sched_obj)
        else:
            raise ValueError
    else:
        return None


def load_w2v(opts):
    import gensim
    w2v_p = Path(opts['w2v']['path']).resolve()
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(str(w2v_p), binary=opts['w2v']['binary'])

    if opts['w2v']['padding_idx'] == 'add':
        w2v_model.add(['@pad@'], [np.zeros(w2v_model.vectors.shape[1])])
        opts['w2v']['padding_idx'] = w2v_model.vocab['@pad@'].index

    opts['w2v']['vocab'] = w2v_model.vocab
    opts['w2v']['weights'] = torch.FloatTensor(w2v_model.vectors)


def load_fasttext(opts):
    import fastText as ft

    fasttext_p = Path(opts['ft']['path']).resolve()
    opts['ft']['model'] = ft.load_model(str(fasttext_p))


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
    model = cls(opts['ms'],
                opts['w2v'],
                opts['ft'],
                opts['elmo'],
                opts['bert'],
                final_drop=opts_model['final_drop'],
                relu=opts_model['relu'])

    return model


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train regression model.')
    arg_parser.add_argument('config_f', help='Path to JSON file with configuration options.')

    cli_args = arg_parser.parse_args()
    config_f = verify_paths(cli_args.config_f)
    with open(config_f, 'r') as config_fh:
        options = json.load(config_fh)

    main(options, config_f)
