import logging
from math import exp
from pathlib import Path
import string
import time
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import torch
from torch import nn
from torch.utils.data import DataLoader

from LazyCharDataset import LazyCharDataset

# Make results reproducible
torch.manual_seed(3)
torch.backends.cudnn.deterministic = True
np.random.seed(3)

# Run all numpy warnings as errors to catch issues with pearsonr
np.seterr(all='raise')


class WordTrainer:
    def __init__(self,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 train_file=None,
                 valid_file=None,
                 test_file=None,
                 batch_size=(64, 64, 64)):
        logging.info(f"Using torch {torch.__version__}")

        self.files = [train_file, valid_file, test_file]
        self.datasets, self.dataloaders = self._set_data_loaders(train_file,
                                                                 valid_file,
                                                                 test_file,
                                                                 batch_size)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_f = None
        self.device = self._set_device()

        self.cat_to_idx = {}
        self.idx_to_cat = {}
        self.all_categories = []
        self.n_categories = 0
        self.categories_from_file()

        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters)

    def categories_from_file(self):
        categories = set()
        for fname in self.files:
            with open(fname, encoding='utf-8') as fhin:
                for line in fhin:
                    category = line.strip().split('\t')[1]
                    categories.add(category)

        self.all_categories = list(categories)
        self.n_categories = len(categories)
        self.cat_to_idx = {cat:idx for idx, cat in enumerate(self.all_categories)}
        self.idx_to_cat = {idx:cat for idx, cat in enumerate(self.all_categories)}


    @staticmethod
    def _set_data_loaders(train_file, valid_file, test_file, batch_size):
        datasets = {
            'train': LazyCharDataset(train_file) if train_file is not None else None,
            'valid': LazyCharDataset(valid_file) if valid_file is not None else None,
            'test': LazyCharDataset(test_file) if test_file is not None else None
        }

        if train_file:
            logging.info(f"Training set size: {len(datasets['train'])}")
        if valid_file:
            logging.info(f"Validation set size: {len(datasets['valid'])}")
        if test_file:
            logging.info(f"Test set size: {len(datasets['test'])}")

        dataloaders = {
            'train': DataLoader(datasets['train'], batch_size=batch_size[0], shuffle=False)
            if train_file is not None else None,
            'valid': DataLoader(datasets['valid'], batch_size=batch_size[1], shuffle=False)
            if valid_file is not None else None,
            'test': DataLoader(datasets['test'], batch_size=batch_size[2], shuffle=False)
            if test_file is not None else None
        }

        return datasets, dataloaders

    @staticmethod
    def _set_device():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            device_id = torch.cuda.current_device()
            logging.info(f"Using GPU {torch.cuda.get_device_name(device_id)}")
        else:
            logging.info('Using CPU')

        return device

    @staticmethod
    def _plot_training(train_losses, valid_losses):
        fig = plt.figure(dpi=300)
        plt.plot(train_losses, label='Training loss')
        plt.plot(valid_losses, label='Validation loss')
        plt.xlabel('epochs')
        plt.legend(frameon=False)
        plt.title('Loss progress')
        plt.show()

        return fig

    def _unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    def letter_to_index(self, letter):
        return self.all_letters.find(letter)

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letter_to_tensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letter_to_index(letter)] = 1
        return tensor

    # Turn a line into a <line_length x n_letters>,
    # or an array of one-hot letter vectors
    def line_to_tensor(self, line, max_len=None):
        if max_len is not None:
            tensor = torch.zeros(max_len, self.n_letters)
        else:
            tensor = torch.zeros(len(line), self.n_letters)

        for letter_idx, letter in enumerate(line):
            tensor[letter_idx][self.letter_to_index(letter)] = 1
        return tensor

    def batch_to_tensor(self, batch):
        batch_max_len = max(map(len, batch))
        sizes = (len(batch), batch_max_len, self.n_letters)
        tensor = torch.empty(sizes)
        for line_idx, line in enumerate(batch):
            tensor[line_idx] = self.line_to_tensor(line, max_len=batch_max_len)
        return tensor

    def categories_to_idx(self, batch):
        return torch.LongTensor([self.cat_to_idx[cat] for cat in batch])

    def idx_to_categories(self, batch):
        return torch.LongTensor([self.idx_to_cat[idx] for idx in batch])

    def prob_and_cat_from_output(self, output):
        m = nn.LogSoftmax(dim=1)
        output = m(output)
        top_category_scores, top_category_idxs = output.topk(1)
        top_category_probs = list(map(exp, top_category_scores.squeeze(dim=0).tolist()))
        top_category_idxs = top_category_idxs.squeeze(dim=0).tolist()
        top_categories = [self.all_categories[idx] for idx in top_category_idxs]

        return list(zip(top_category_probs, top_categories))

    def train(self, epochs=10, checkpoint_f='checkpoint.pth', log_update_freq=0, patience=0):
        """ log_update_freq: show a log message every X percent in a batch's progress.
            E.g. for a value of 25, 4 messages will be printed per batch (100/25=4)
        """
        logging.info('Training started.')
        train_start = time.time()

        self.checkpoint_f = checkpoint_f

        valid_loss_min = np.inf
        train_losses, valid_losses = [], []
        last_saved_epoch = 0
        # keep
        total_train_time = 0
        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss = self._process('train', log_update_freq, epoch)
            total_train_time += time.time() - epoch_start

            # Calculate average losses
            train_loss = np.mean(train_loss)
            train_losses.append(train_loss)

            # VALIDATION
            valid_loss = self._process('valid', log_update_freq, epoch)

            valid_loss = np.mean(valid_loss)
            valid_losses.append(valid_loss)

            # Log epoch statistics
            logging.info(f"Epoch {epoch} - completed in {(time.time() - epoch_start):.0f} seconds"
                         f"\nTraining Loss: {train_loss:.6f}"
                         f"\nValidation loss: {valid_loss:.6f}")

            # Save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                logging.info(f'!! Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).')
                logging.info(f'!! Saving model as {self.checkpoint_f}...')

                torch.save(self.model.state_dict(), self.checkpoint_f)
                last_saved_epoch = epoch
                valid_loss_min = valid_loss
            else:
                logging.info(
                    f"!! Valid loss not improved. (Min. = {valid_loss_min}; last save at ep. {last_saved_epoch})")
                if train_loss <= valid_loss:
                    logging.warning(f"!! Training loss is lte validation loss. Might be overfitting!")

            # Early-stopping
            if patience:
                if (epoch - last_saved_epoch) == patience:
                    logging.info(f"Stopping early at epoch {epoch} (patience={patience})...")
                    break

            # Optimise with scheduler
            if self.scheduler is not None:
                self.scheduler.step(valid_loss)

        fig = self._plot_training(train_losses, valid_losses)

        logging.info(f"Training completed in {(time.time() - train_start):.0f} seconds"
                     f"\nMin. valid loss: {valid_loss_min}\nLast saved epoch: {last_saved_epoch}"
                     f"\nPerformance: {len(self.datasets['train']) // total_train_time:.0f} sentences/s")

        return self.checkpoint_f, fig

    def _process(self, do, log_update_freq, epoch=None):
        if do not in ('train', 'valid', 'test'):
            raise ValueError("Use 'train', 'valid', or 'test' for 'do'.")

        losses = np.array([])

        self.model = self.model.to(self.device)
        if do == 'train':
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        if log_update_freq:
            nro_batches = len(self.datasets[do]) // self.dataloaders[do].batch_size
            update_interval = nro_batches * (log_update_freq / 100)
            update_checkpoints = {int(nro_batches - (i * update_interval)) for i in range((100 // log_update_freq))}

        for batch_idx, data in enumerate(self.dataloaders[do], 1):
            # 0. Clear gradients
            if do == 'train':
                self.optimizer.zero_grad()

            # 1. Data prep
            names, categories = data
            cat_idx_tensor = self.categories_to_idx(categories).to(self.device)
            names_tensor = self.batch_to_tensor(names).to(self.device)

            # 2. Predictions
            pred = self.model(names_tensor)
            loss = self.criterion(pred, cat_idx_tensor)

            # 3. Optimise during training
            if do == 'train':
                loss.backward()
                self.optimizer.step()

            # 4. Save results
            losses = np.append(losses, float(loss))

            if log_update_freq and batch_idx in update_checkpoints:
                if do in ('train', 'valid'):
                    logging.info(f"{do.capitalize()} epoch {epoch}, batch nr. {batch_idx}/{nro_batches}...")
                else:
                    logging.info(f"{do.capitalize()}, batch nr. {batch_idx}/{nro_batches}...")

        return losses

    def test(self, checkpoint_f='checkpoint.pth', log_update_freq=0):
        logging.info('Testing started.')
        test_start = time.time()

        if self.checkpoint_f is None:
            self.model.load_state_dict(torch.load(checkpoint_f, map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(self.checkpoint_f, map_location=self.device))

        test_loss = self._process('test', log_update_freq)
        test_loss = np.mean(test_loss)

        logging.info(f"Testing completed in {(time.time() - test_start):.0f} seconds"
                     f"\nLoss: {test_loss:.6f}")

        return test_loss

    def predict(self, checkpoint_f='checkpoint.pth'):
        if self.checkpoint_f is None:
            self.model.load_state_dict(torch.load(checkpoint_f, map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(self.checkpoint_f, map_location=self.device))
        self.model = self.model.to(self.device)

        with torch.no_grad():
            while True:
                person_name = input('Name: ')
                names_tensor = self.batch_to_tensor([person_name]).to(self.device)
                pred = self.model(names_tensor)
                print(pred)
                print(self.prob_and_cat_from_output(pred))
