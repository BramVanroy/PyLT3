import logging
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence


class RegressionRNN(nn.Module):
    def __init__(self, ms, w2v, elmo, bert, final_drop, relu):
        super(RegressionRNN, self).__init__()
        self.ms = ms
        self.w2v = w2v
        self.elmo = elmo
        self.bert = bert
        self.final_drop = final_drop
        self.relu = relu

        input_fc = 0

        if ms['use']:
            self.ms_rlayer, input_fc = RegressionRNN._build_recurrent_layer(ms, input_fc)
            logging.info('Morphosyntactic features enabled...')

        if w2v['use']:
            self.w2v_layer = nn.Embedding.from_pretrained(w2v['weights'], freeze=w2v['freeze'])
            self.w2v_rlayer, input_fc = RegressionRNN._build_recurrent_layer(w2v, input_fc)
            logging.info('Word embeddings enabled...')

        if elmo['use']:
            from allennlp.modules.elmo import Elmo

            self.elmo_layer = Elmo(elmo['options_path'], elmo['weights_path'], 1, dropout=elmo['dropout'])

            if elmo['linear_layer'] is not None:
                self.elmo_llayer = nn.Linear(elmo['dim'], elmo['linear_layer']['dim'])
                input_fc += elmo['linear_layer']['dim']
            else:
                input_fc += elmo['dim']
            logging.info('ELMo enabled...')

        if bert['use']:
            from pytorch_pretrained_bert.modeling import BertModel

            self.bert_layer = BertModel.from_pretrained(bert['name'])
            if bert['freeze'] and bert['freeze'] in {'embeddings', 'encoder', 'all'}:
                # Freeze all params, embeddings, or encoder
                for name, param in self.bert_layer.named_parameters():
                    if bert['freeze'] == 'all':
                        param.requires_grad = False
                    elif name.startswith(bert['freeze']):
                        param.requires_grad = False

            if bert['linear_layer'] is not None:
                self.bert_llayer = nn.Linear(bert['dim'] * len(bert['concat_layers']),
                                             bert['linear_layer']['dim'])
                input_fc += elmo['linear_layer']['dim']
            else:
                input_fc += elmo['dim']
            logging.info('Bert enabled...')

        self.dropout_layer = nn.Dropout(final_drop) if final_drop > 0 else None
        self.linear_layer = nn.Linear(input_fc, 1)

        if relu['use']:
            if 'negative_slope' in relu and relu['negative_slope']:
                neg_slope = relu['negative_slope']
            else
                neg_slope = 0.01
            self.relu_layer = nn.LeakyReLU(negative_slope=neg_slope) if relu['type'] == 'leaky' else nn.ReLU()

    def forward(self, ms_input=None, w2v_ids=None, elmo_ids=None, bert_input=None):
        if self.ms['use'] and ms_input is not None:
            packed_ms_out, _ = self.ms_rlayer(ms_input)
            unpacked_ms_out, ms_lengths = pad_packed_sequence(packed_ms_out, batch_first=True)

            if self.ms['recurrent_layer']['bidirectional']:
                unpacked_ms_out = unpacked_ms_out[:, :, :self.hidden_dim] + unpacked_ms_out[:, :, self.hidden_dim:]

            # Get last item of each sequence, based on their *actual* lengths
            final_ms = unpacked_ms_out[torch.arange(len(ms_lengths)), ms_lengths - 1, :]
        else:
            final_ms = None

        if self.w2v['use'] and w2v_ids is not None:
            w2v_out = self.w2v_layer(w2v_ids)
            w2v_out, _ = self.w2v_rlayer(w2v_out)

            if self.w2v['recurrent_layer']['bidirectional']:
                w2v_out = w2v_out[:, :, :self.hidden_dim] + w2v_out[:, :, self.hidden_dim:]

            final_w2v = w2v_out[:, -1, :]
        else:
            final_w2v = None

        if self.elmo['use'] and elmo_ids is not None:
            elmo_out = self.elmo_layer(elmo_ids)
            elmo_out = elmo_out['elmo_representations'][0]
            final_elmo = elmo_out[:, -1, :]

            if self.elmo['linear_layer'] is not None:
                final_elmo = self.elmo_llayer(final_elmo)
        else:
            final_elmo = None

        if self.bert['use'] and bert_input is not None:
            bert_ids, bert_mask = bert_input

            all_bert_layers, _ = self.bert_layer(bert_ids, attention_mask=bert_mask)
            bert_concat = torch.cat(tuple([all_bert_layers[i] for i in self.bert_layers]), dim=-1)
            # Pooling by also setting masked items to zero
            bert_mask = torch.FloatTensor(bert_mask).unsqueeze(2)
            # Multiply output with mask to only retain non-paddding tokens
            bert_pooled = torch.mul(bert_concat, bert_mask)

            # First item ['CLS'] is sentence representation
            final_bert = bert_pooled[:, 0, :]

            if self.bert['linear_layer'] is not None:
                final_bert = self.bert_llayer(final_bert)
        else:
            final_bert = None

        sentence_finals = tuple([final for final in [final_w2v, final_elmo, final_bert] if final is not None])

        # Sentence features concatenate
        if len(sentence_finals) > 1:
            sentence_cat = torch.cat(sentence_finals, dim=1)
        elif len(sentence_finals) == 1:
            sentence_cat = sentence_finals[0]
        else:
            sentence_cat = None

        # Sentence features + MS features concatenate
        if final_ms is not None and sentence_cat is not None:
            sentence_ms_cat = torch.cat((sentence_cat, final_ms), dim=1)
        elif final_ms is not None:
            sentence_ms_cat = final_ms
        elif sentence_cat is not None:
            sentence_ms_cat = sentence_cat

        # Only use the last item's output
        if self.final_drop > 0:
            sentence_ms_cat = self.dropout_layer(sentence_ms_cat)

        regression = self.linear(sentence_ms_cat)

        if self.relu['use']:
            regression = self.lrelu(regression)

        return regression

    @staticmethod
    def _build_recurrent_layer(opt, input_fc):
        if opt['recurrent_layer'] is not None:
            rlayer_type = nn.GRU if opt['recurrent_layer']['type'] == 'gru' else nn.LSTM
            rlayer = rlayer_type(opt['dim'],
                                 opt['recurrent_layer']['dim'],
                                 bidirectional=opt['recurrent_layer']['bidirectional'],
                                 batch_first=True)
            input_fc += opt['recurrent_layer']['dim']
        else:
            rlayer = None
            input_fc += opt['dim']

        return rlayer, input_fc
