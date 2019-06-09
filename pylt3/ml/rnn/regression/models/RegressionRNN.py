import logging
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence


class RegressionRNN(nn.Module):
    def __init__(self, ms, w2v, ft, elmo, bert, final_drop, relu):
        super(RegressionRNN, self).__init__()
        self.ms = ms
        self.w2v = w2v
        self.ft = ft
        self.elmo = elmo
        self.bert = bert
        self.final_drop = final_drop
        self.relu = relu

        groups = []
        if ms['use']:
            self.ms_group = MorphosynGroup(ms)
            groups.append(self.ms_group)
            logging.info('Morphosyntactic features enabled...')

        if w2v['use']:
            self.w2v_group = Word2VecGroup(w2v)
            groups.append(self.w2v_group)
            logging.info('Word embeddings enabled...')

        if ft['use']:
            self.ft_group = FastTextGroup(ft)
            groups.append(self.ft_group)
            logging.info('fastText enabled...')

        if elmo['use']:
            self.elmo_group = ElmoGroup(elmo)
            groups.append(self.elmo_group)
            logging.info('ELMo enabled...')

        if bert['use']:
            self.bert_group = BertGroup(bert)
            groups.append(self.bert_group)
            logging.info('Bert enabled...')

        self.concat = Concatenator()

        self.dropout_layer = nn.Dropout(final_drop) if final_drop > 0 else None
        self.linear_layer = nn.Linear(sum([g.out_dims for g in groups]), 1)

        if relu['use']:
            if 'negative_slope' in relu and relu['negative_slope']:
                neg_slope = relu['negative_slope']
            else:
                neg_slope = 0.001
            self.relu_layer = nn.LeakyReLU(negative_slope=neg_slope) if relu['type'] == 'leaky' else nn.ReLU()

    def forward(self, ms_input=None, w2v_ids=None, ft_vec=None, elmo_ids=None, bert_input=None):
        final_ms = self.ms_group(ms_input) if self.ms['use'] and ms_input is not None else None

        final_w2v = self.w2v_group(w2v_ids) if self.w2v['use'] and w2v_ids is not None else None
        final_ft = self.ft_group(ft_vec) if self.ft['use'] and ft_vec is not None else None
        final_elmo = self.elmo_group(elmo_ids) if self.elmo['use'] and elmo_ids is not None else None
        final_bert = self.bert_group(bert_input) if self.bert['use'] and bert_input is not None else None
        sentence_finals = tuple([final for final in [final_w2v, final_ft, final_elmo, final_bert]
                                 if final is not None])

        sentence_cat = self.concat(sentence_finals) if len(sentence_finals) > 0 else None

        sentence_ms_cat = self.concat(tuple([final for final in [final_ms, sentence_cat] if final is not None]))

        if self.final_drop > 0:
            sentence_ms_cat = self.dropout_layer(sentence_ms_cat)

        regression = self.linear_layer(sentence_ms_cat)

        if self.relu['use']:
            regression = self.relu_layer(regression)

        return regression


class Concatenator(nn.Module):
    def __init__(self):
        super(Concatenator, self).__init__()

    def forward(self, inputs):
        if len(inputs) > 1:
            sentence_cat = torch.cat(inputs, dim=1)
        else:
            sentence_cat = inputs[0]

        return sentence_cat


class RNNLayerGroup(nn.Module):
    def __init__(self):
        super(RNNLayerGroup, self).__init__()
        self.out_dims = 0

    def _build_recurrent_layer(self, opts):
        if opts['recurrent_layer'] is not None:
            rec_opts = opts['recurrent_layer']
            rlayer_type = nn.GRU if opts['recurrent_layer']['type'] == 'gru' else nn.LSTM
            rlayer = rlayer_type(opts['dim'],
                                 rec_opts['dim'],
                                 dropout=rec_opts['dropout'] if 'dropout' in rec_opts else 0,
                                 num_layers=rec_opts['num_layers'] if 'num_layers' in rec_opts else 1,
                                 bidirectional=rec_opts['bidirectional'],
                                 batch_first=True)
            self.out_dims = rec_opts['dim']
        else:
            rlayer = None
            self.out_dims = opts['dim']

        return rlayer

    def forward(self, *input):
        raise NotImplementedError


class MorphosynGroup(RNNLayerGroup):
    def __init__(self, opts):
        super(MorphosynGroup, self).__init__()

        self.opts = opts
        self.rlayer = self._build_recurrent_layer(opts)

    def forward(self, ms_input):
        ms_dim = self.opts['recurrent_layer']['dim']
        packed_ms_out, _ = self.rlayer(ms_input)
        # print('packed_ms_out after gru', packed_ms_out.size())
        unpacked_ms_out, ms_lengths = pad_packed_sequence(packed_ms_out, batch_first=True)
        # print('unpacked_ms_out', unpacked_ms_out.size())

        if self.opts['recurrent_layer']['bidirectional']:
            unpacked_ms_out = unpacked_ms_out[:, :, :ms_dim] + unpacked_ms_out[:, :, ms_dim:]
            # print('unpacked_ms_out after bidirectional', unpacked_ms_out.size())

        # Get last item of each sequence, based on their *actual* lengths
        final_ms = unpacked_ms_out[torch.arange(len(ms_lengths)), ms_lengths - 1, :]
        # print('final_ms', final_ms.size())

        return final_ms


class Word2VecGroup(RNNLayerGroup):
    def __init__(self, opts):
        super(Word2VecGroup, self).__init__()

        self.opts = opts
        self.embedding = nn.Embedding\
            .from_pretrained(opts['weights'], freeze=opts['freeze'], padding_idx=opts['padding_idx'])
        self.rlayer = self._build_recurrent_layer(opts)

    def forward(self, w2v_ids):
        w2v_dim = self.opts['recurrent_layer']['dim']
        w2v_embed = self.embedding(w2v_ids)
        w2v_out, _ = self.rlayer(w2v_embed)

        if self.opts['recurrent_layer']['bidirectional']:
            w2v_out = w2v_out[:, :, :w2v_dim] + w2v_out[:, :, w2v_dim:]

        final_w2v = w2v_out[:, -1, :]

        return final_w2v


class FastTextGroup(RNNLayerGroup):
    def __init__(self, opts):
        super(FastTextGroup, self).__init__()

        self.opts = opts
        self.rlayer = self._build_recurrent_layer(opts)

    def forward(self, ft_vec):
        ft_dim = self.opts['recurrent_layer']['dim']
        packed_ft_out, _ = self.rlayer(ft_vec)

        unpacked_ft_out, ft_lengths = pad_packed_sequence(packed_ft_out, batch_first=True)

        if self.opts['recurrent_layer']['bidirectional']:
            unpacked_ft_out = unpacked_ft_out[:, :, :ft_dim] + unpacked_ft_out[:, :, ft_dim:]

        # Get last item of each sequence, based on their *actual* lengths
        final_ft = unpacked_ft_out[torch.arange(len(ft_lengths)), ft_lengths - 1, :]

        return final_ft


class ElmoGroup(RNNLayerGroup):
    def __init__(self, opts):
        from allennlp.modules.elmo import Elmo
        super(ElmoGroup, self).__init__()

        self.opts = opts
        self.elmo_layer = Elmo(opts['options_path'], opts['weights_path'], 1, dropout=opts['dropout'])

        if 'linear_layer' in opts and opts['linear_layer'] is not None:
            self.elmo_llayer = nn.Linear(opts['dim'], opts['linear_layer']['dim'])
            self.out_dims = opts['linear_layer']['dim']
        else:
            self.out_dims = opts['dim']

    def forward(self, elmo_ids):
        elmo_out = self.elmo_layer(elmo_ids)
        # Only using one representation, so get it by first index
        elmo_out = elmo_out['elmo_representations'][0]
        final_elmo = elmo_out[:, -1, :]

        if 'linear_layer' in self.opts and self.opts['linear_layer'] is not None:
            final_elmo = self.elmo_llayer(final_elmo)

        return final_elmo


class BertGroup(RNNLayerGroup):
    def __init__(self, opts):
        from pytorch_pretrained_bert.modeling import BertModel
        super(BertGroup, self).__init__()

        self.opts = opts

        self.bert_layer = BertModel.from_pretrained(opts['name'])
        if opts['freeze'] and opts['freeze'] in {'embeddings', 'encoder', 'all'}:
            # Freeze all params, embeddings, or encoder
            for name, param in self.bert_layer.named_parameters():
                if opts['freeze'] == 'all':
                    param.requires_grad = False
                elif name.startswith(opts['freeze']):
                    param.requires_grad = False

        if opts['linear_layer'] is not None:
            self.bert_llayer = nn.Linear(opts['dim'] * len(opts['concat_layers']),
                                         opts['linear_layer']['dim'])
            self.out_dims = opts['linear_layer']['dim']
        else:
            self.out_dims = opts['dim']

    def forward(self, bert_input):
        bert_ids, bert_mask = bert_input

        all_bert_layers, _ = self.bert_layer(bert_ids, attention_mask=bert_mask)
        bert_concat = torch.cat(tuple([all_bert_layers[i] for i in self.bert['concat_layers']]), dim=-1)
        # Pooling by also setting masked items to zero
        bert_mask = bert_mask.unsqueeze(2)
        # Multiply output with mask to only retain non-paddding tokens
        bert_pooled = torch.mul(bert_concat, bert_mask)

        # First item ['CLS'] is sentence representation
        final_bert = bert_pooled[:, 0, :]

        if self.bert['linear_layer'] is not None:
            final_bert = self.bert_llayer(final_bert)

        return final_bert
