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


def main(old_opts):
    for lr in (1E-03, 1E-04, 1E-05):
        for dropout in (0, 0.2, 0.4):
            for hidden_dim in (256, 384, 512):
                opts = deepcopy(old_opts)

                output_p = Path(opts['trainer'].pop('output_dir')).resolve()

                for o in (opts, old_opts):
                    # o['model']['relu']['use'] = relu
                    o['optimizer']['lr'] = lr
                    o['model']['final_drop'] = dropout
                    # o['ms']['recurrent_layer']['dropout'] = dropout
                    # o['w2v']['recurrent_layer']['dropout'] = dropout
                    # o['ft']['recurrent_layer']['dropout'] = dropout
                    o['elmo']['recurrent_layer']['dropout'] = dropout
                    o['elmo']['dropout'] = dropout

                    # o['ms']['recurrent_layer']['dim'] = hidden_dim
                    # o['w2v']['recurrent_layer']['dim'] = hidden_dim
                    # o['ft']['recurrent_layer']['dim'] = hidden_dim
                    o['elmo']['recurrent_layer']['dim'] = hidden_dim

                    # o['ms']['pretrained']['freeze'] = freeze
                    # o['w2v']['pretrained']['freeze'] = freeze

                    # o['bert']['linear_layer']['dim'] = linear_dim

                set_seed()

                if opts['w2v']['use']:
                    load_w2v(opts)

                if opts['ft']['use']:
                    load_fasttext(opts)

                model = load_model(opts)

                if opts['ms']['use'] and opts['ms']['pretrained']['use']:
                    model = load_weights('ms', opts['ms']['pretrained'], model)

                if opts['w2v']['use'] and opts['w2v']['pretrained']['use']:
                    model = load_weights('w2v', opts['w2v']['pretrained'], model)

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

                # write config file based on actual values
                with open(config_out, 'w') as fhout:
                    json.dump(old_opts, fhout)

                # move output model to output_dir
                move(best_model_p, model_out)
                # Save plot
                fig.savefig(output_p.joinpath(s+'plot.png'))


def load_weights(place, opts, model):
    """ https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2 """
    p_weights = Path(opts['path']).resolve()

    pre_state = torch.load(p_weights)['model_state_dict']
    model_state = model.state_dict()
    pre_state = {k: v for k, v in pre_state.items() if k in model_state and k.startswith(place)}
    model_state.update(pre_state)
    model.load_state_dict(model_state)

    # freeze pretrained layers?
    if opts['freeze']:
        for name, param in model.named_parameters():
            if name.startswith(place):
                param.requires_grad = False
                print(f"{name}: frozen...")

    return model


def get_output_prefix(loss, pearson, optim_name, opts):
    s = f"loss{loss:.2f}-pearson{pearson:.2f}-"
    s += f"{optim_name}-lr{opts['optimizer']['lr']:.0E}-"

    used = []
    for feat in ('ms', 'w2v', 'ft'):
        if opts[feat]['use']:
            feat_s = f"{opts[feat]['dim']}{feat}-{opts[feat]['recurrent_layer']['type']}-"
            feat_s += f"h{opts[feat]['recurrent_layer']['dim']}"
            if opts[feat]['recurrent_layer']['bidirectional']:
                feat_s += '-bi'
            if opts[feat]['pretrained']['use']:
                feat_s += '-pre'
                if opts[feat]['pretrained']['freeze']:
                    feat_s += '-frozen'
            used.append(feat_s)

    if opts['elmo']['use']:
        feat_s = f"elmo-{opts['elmo']['recurrent_layer']['type']}-"
        feat_s += f"h{opts['elmo']['recurrent_layer']['dim']}"
        if opts['elmo']['recurrent_layer']['bidirectional']:
            feat_s += '-bi'

        used.append(feat_s)

    if opts['bert']['use']:
        bert_s = f"{opts['bert']['dim']}bert"
        bert_s += f"-llayer{opts['bert']['linear_layer']['dim']}"
        used.append(bert_s)

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
    optim_copy = deepcopy(optim_obj)
    optim_name = optim_copy.pop('name')
    if optim_name.lower() == 'bert-adam':
        from pytorch_pretrained_bert.optimization import BertAdam
        return BertAdam([p for p in model.parameters() if p.requires_grad],
                        **optim_copy), optim_name
    elif optim_name.lower() == 'adam':
        return optim.Adam([p for p in model.parameters() if p.requires_grad],
                          **optim_copy), optim_name
    else:
        raise NotImplementedError('This optimiser has not been implemented in this system yet.')


def get_scheduler(sched_obj, optimizer):
    sched_copy = deepcopy(sched_obj)
    if sched_copy.pop('use'):
        sched_name = sched_copy.pop('name')
        if sched_name.lower() == 'reducelronplateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **sched_copy)
        else:
            raise NotImplementedError('This scheduler has not been implemented in this system yet.')
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

    main(options)