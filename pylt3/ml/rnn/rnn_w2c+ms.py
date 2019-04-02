from pathlib import Path
import time

import numpy as np
import torch
import visdom
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gensim

from scipy.stats import pearsonr

import sys
sys.path.append(r'C:\Python\projects\PyLT3')

from pylt3.ml.rnn.LazyTextDataset import LazyTextDataset

# vis = visdom.Visdom()

# Load embeddings
embed_p = Path('..\..\..\data\dpc\ml\other\dpc+news2017.dim146-ep10-min2-win10-repl.w2v_model').resolve()
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(str(embed_p))
# add a padding token with only zeros
w2v_model.add(['@pad@'], [np.zeros(w2v_model.vectors.shape[1])])
embed_weights = torch.FloatTensor(w2v_model.vectors)

class LSTMRegressor(nn.Module):
    def __init__(self, hidden_dim, ms_dim=None, embeddings=None):
        super(LSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim

        if embeddings is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(embeddings)
            embed_size = embeddings.shape[1]
            self.word_embeddings.weight.requires_grad = False

            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.
            self.w2v_lstm = nn.LSTM(embed_size, hidden_dim, bidirectional=True)
            self.hc_w2v = None

        if ms_dim is not None:
            self.ms_lstm = nn.LSTM(ms_dim, hidden_dim, bidirectional=True)
            self.hc_ms = None

        # The linear layer that maps from hidden state space to a single output
        self.linear = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size, device):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, batch_size, self.hidden_dim).to(device),
                torch.zeros(2, batch_size, self.hidden_dim).to(device))

    def forward(self, batch_size, sentence_input=None, ms_input=None):
        if sentence_input is not None:
            embeds = self.word_embeddings(sentence_input)
            w2v_out, self.hc_w2v = self.w2v_lstm(embeds.view(-1, batch_size, embeds.size(2)), self.hc_w2v)

            w2v_first_bi = w2v_out[:, :, :self.hidden_dim]
            w2v_last_bi = w2v_out[:, :, self.hidden_dim:]
            w2v_sum_bi = w2v_first_bi + w2v_last_bi

        if ms_input is not None:
            input_size = ms_input.size(1)
            ms_out, self.hc_ms = self.ms_lstm(ms_input.view(-1, batch_size, input_size), self.hc_ms)

            ms_first_bi = ms_out[:, :, :self.hidden_dim]
            ms_last_bi = ms_out[:, :, self.hidden_dim:]
            ms_sum_bi = ms_first_bi + ms_last_bi

        if not (sentence_input is None or ms_input is None):
            summed = torch.cat((w2v_sum_bi, ms_sum_bi))

        elif ms_input is None:
            summed = w2v_sum_bi
        else:
            summed = ms_sum_bi

        # Only use the last item's output
        summed = summed[-1, :, :]

        regression = F.relu(self.linear(summed))

        return regression


class RegressionRNN:
    def __init__(self, train_files=None, test_files=None, dev_files=None):
        print('Using torch ' + torch.__version__)

        self.datasets, self.dataloaders = RegressionRNN._set_data_loaders(train_files, test_files, dev_files)
        self.device = RegressionRNN._set_device()

        self.model = self.w2v_vocab = self.criterion = self.optimizer = self.scheduler = self.checkpoint_f = self.class_to_idx = None

    @staticmethod
    def _set_data_loaders(train_files, test_files, dev_files):
        RegressionRNN._verify_input(train_files, test_files, dev_files)

        # labels first
        datasets = {
            'train': LazyTextDataset(train_files) if train_files is not None else None,
            'test': LazyTextDataset(test_files) if test_files is not None else None,
            'dev': LazyTextDataset(dev_files) if dev_files is not None else None
        }

        # h5py doesn't allow to load with multiple workers. Don't use num_workers
        dataloaders = {
            'train': DataLoader(datasets['train'], batch_size=64, shuffle=True) if train_files is not None else None,
            'test': DataLoader(datasets['test'], batch_size=1) if test_files is not None else None,
            'dev': DataLoader(datasets['dev'], batch_size=1) if dev_files is not None else None
        }

        return datasets, dataloaders

    @staticmethod
    def _set_device():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            device_id = torch.cuda.current_device()
            print(f'Using GPU {torch.cuda.get_device_name(device_id)}...')
            # print('Memory Usage:')
            # print('Allocated:', round(torch.cuda.memory_allocated(device_id) / 1024 ** 3, 1), 'GB')
            # print('Cached:   ', round(torch.cuda.memory_cached(device_id) / 1024 ** 3, 1), 'GB')
        else:
            print('Using CPU...')

        return device

    @staticmethod
    def _verify_input(train_files, test_files, dev_files):
        for f_kind in [train_files, test_files, dev_files]:
            if f_kind is None:
                continue

            for f in f_kind:
                print(f)
                if not Path(f).resolve().is_file():
                    raise ValueError(f"Input file {str(f)} does not exist.")

    @staticmethod
    def prepare_lines(data, split_on=None, cast_to=None, min_size=None, pad_str=None, max_size=None, to_numpy=False):
        out = []
        for line in data:
            line = line.strip()
            if split_on:
                line = line.split(split_on)
                line = list(filter(None, line))
            else:
                line = [line]

            if cast_to is not None:
                line = [cast_to(l) for l in line]

            if min_size is not None and len(line) < min_size:
                line += (min_size - len(line)) * ['@pad@']
            elif max_size and len(line) > max_size:
                line = line[:max_size]

            if to_numpy:
                line = np.array(line)

            out.append(line)

        if to_numpy:
            out = np.array(out)

        return out

    @staticmethod
    def get_pearson(i, j):
        return pearsonr(i, j)

    def prepare_w2v(self, data):
        idxs = []
        for seq in data:
            tok_idxs = []
            for word in seq:
                try:
                    tok_idxs.append(self.w2v_vocab[word].index)
                except KeyError:
                    tok_idxs.append(self.w2v_vocab['@unk@'].index)
            idxs.append(tok_idxs)
        idxs = torch.tensor(idxs, dtype=torch.long)

        return idxs

    def train(self, epochs=10, checkpoint_f='checkpoint.pth'):
        self.checkpoint_f = checkpoint_f

        train_size = len(self.dataloaders['train'].dataset)
        train_batch_size = self.dataloaders['train'].batch_size

        # ADD VALID AND TEST

        valid_loss_min = np.Inf
        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss, train_results = self._train(epoch)

            pearson_corr = self.get_pearson(train_results['predictions'], train_results['targets'])
            # valid_correct, valid_loss = self._validate()

            # calculate average losses
            train_loss = np.mean(train_loss)
            # valid_loss = valid_loss / valid_size

            train_losses.append(train_loss)
            # valid_losses.append(valid_loss)

            # print training/validation statistics
            # print(f'----------\n'
            #      f'Epoch {epoch} - completed in {(time.time() - epoch_start):.0f} seconds\n'
            #      f'Training Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}\n'
            #      f'Training Accuracy: {train_acc:.2f}%\tValidation Accuracy: {valid_acc:.2f}%')

            print(f'----------\n'
                  f'Epoch {epoch} - completed in {(time.time() - epoch_start):.0f} seconds\n'
                  f'Training Loss: {train_loss:.6f}\t Pearson: {pearson_corr}')

            """

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min and train_loss > valid_loss:
                print(f'!! Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...')

                torch.save(self.model.state_dict(), self.checkpoint_f)
                valid_loss_min = valid_loss

            if train_loss <= valid_loss:
                print('!! Training loss is lte validation loss. Might be overfitting!')

            if self.scheduler is not None:
                self.scheduler.step(valid_loss)
            """
        # RegressionRNN._plot_training(train_losses, valid_losses, train_accs, valid_accs)
        print('Done training...')

    def _train(self, epoch, vis_win=None):
        train_results = {'predictions': [], 'targets': []}
        train_loss = []
        train_batch_size = len(self.dataloaders['train'].dataset) // self.dataloaders['train'].batch_size

        self.model = self.model.to(self.device)
        self.model.train()
        for batch_idx, data in enumerate(self.dataloaders['train'], 1):
            sentence = data[0]
            ms = data[1]
            target = data[-1]

            # utils, split_on=None, cast_to=None, min_size=None, pad_str=None max_size=None, numpy_it=False
            sentence = self.prepare_lines(sentence, split_on=' ', min_size=100, max_size=100)
            sent_w2v_idxs = self.prepare_w2v(sentence)

            ms = torch.Tensor(self.prepare_lines(ms, split_on=' ', cast_to=int))

            target = torch.Tensor(self.prepare_lines(target, cast_to=float))
            train_results['targets'].extend(target.tolist())

            curr_batch_size = target.size(0)

            sent_w2v_idxs, ms, target = sent_w2v_idxs.to(self.device), ms.to(self.device), target.to(self.device)
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            self.model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            self.model.hc_w2v = self.model.init_hidden(curr_batch_size, self.device)
            self.model.hc_ms = self.model.init_hidden(curr_batch_size, self.device)
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.

            # Step 3. Run our forward pass.
            pred = self.model(curr_batch_size, sentence_input=sent_w2v_idxs, ms_input=ms)
            print(target)
            print(pred)
            exit(0)
            train_results['predictions'].extend(pred.tolist())

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = self.criterion(pred, target)

            if batch_idx % 250 == 0:
                print(f"EPOCH {epoch}: Batch {batch_idx}/{train_batch_size}. Loss: {float(loss)}")
            # must change this to minimise memory footprint
            loss.backward()
            self.optimizer.step()

            train_loss.append(float(loss))

        return train_loss, train_results

    def _validate(self):
        valid_loss = 0.0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.dataloaders['valid']:
                data, target = data.to(self.device), target.to(self.device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                # calculate the batch loss
                loss = self.criterion(output, target)
                # update average validation loss
                valid_loss += loss.item() * data.size(0)

                # get accuracy
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()

        return correct, valid_loss

    def test(self, checkpoint_f='checkpoint.pth'):
        try:
            if self.checkpoint_f is None:
                checkpoint = torch.load(checkpoint_f)
            else:
                checkpoint = torch.load(self.checkpoint_f)
            self.model.load_state_dict(checkpoint['state_dict'])
        except KeyError:
            # In earlier versions I only saved the state_dict - without a wrapper dictionary
            if self.checkpoint_f is None:
                self.model.load_state_dict(torch.load(checkpoint_f))
            else:
                self.model.load_state_dict(torch.load(self.checkpoint_f))

        correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.dataloaders['test']:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()

        test_size = len(self.dataloaders['test'].dataset)
        accuracy = 100 * correct / test_size

        print(f'Testing accuracy: {accuracy:.2f}% on {test_size} images.')

HIDDEN_DIM = 200
MS_SIZE = 102

regr = RegressionRNN(train_files=(r'C:\Python\projects\PyLT3\data\dpc\prep\train.tok.low.en',
                                  r'C:\Python\projects\PyLT3\data\dpc\ml\train\dpc.train.feats.txt',
                                  r'C:\Python\projects\PyLT3\data\dpc\prep\train.cross'))
regr.w2v_vocab = w2v_model.vocab
regr.model = LSTMRegressor(HIDDEN_DIM, MS_SIZE, embed_weights)
# regr.model = LSTMRegressor(HIDDEN_DIM, MS_SIZE)
regr.criterion = nn.MSELoss()
regr.optimizer = optim.Adam(filter(lambda p: p.requires_grad, regr.model.parameters()))
regr.scheduler = optim.lr_scheduler.ReduceLROnPlateau(regr.optimizer, 'min', factor=0.1, patience=5, verbose=True)

regr.train()