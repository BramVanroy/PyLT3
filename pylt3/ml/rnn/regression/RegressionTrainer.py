import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence
from torch.utils.data import DataLoader

from allennlp.modules.elmo import batch_to_ids
from pytorch_pretrained_bert.tokenization import BertTokenizer


from ..LazyTextDataset import LazyTextDataset

# Make results reproducible
torch.manual_seed(3)
torch.backends.cudnn.deterministic = True
np.random.seed(3)

# Run all numpy warnings as errors to catch issues with pearsonr
np.seterr(all='raise')


class RegressionTrainer:
    def __init__(self,
                 ms,
                 w2v,
                 fasttext,
                 elmo,
                 bert,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 train_files=None,
                 valid_files=None,
                 test_files=None,
                 batch_size=(64, 64, 64)):
        logging.info(f"Using torch {torch.__version__}")

        self.datasets, self.dataloaders = self._set_data_loaders(train_files,
                                                                 valid_files,
                                                                 test_files,
                                                                 batch_size)
        self.batch_size = batch_size
        self.device = self._set_device()

        self.ms = ms
        self.w2v = w2v
        self.fasttext = fasttext
        self.elmo = elmo

        self.bert = bert
        self.bert['tokenizer'] = self.init_bert_tokenizer(bert) if bert['use'] else None

        self.use_sentences = any((w2v['use'], fasttext['use'], elmo['use'], bert['use']))

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_f = None

    @staticmethod
    def init_bert_tokenizer(bert):
        # If model is uncased, do_lower_case
        do_lower_case = bert['name'].endswith('-uncased')

        return BertTokenizer.from_pretrained(bert['name'], do_lower_case=do_lower_case)

    @staticmethod
    def _set_data_loaders(train_files, valid_files, test_files, batch_size):
        datasets = {
            'train': LazyTextDataset(train_files) if train_files is not None else None,
            'valid': LazyTextDataset(valid_files) if valid_files is not None else None,
            'test': LazyTextDataset(test_files) if test_files is not None else None
        }

        if train_files:
            logging.info(f"Training set size: {len(datasets['train'])}")
        if valid_files:
            logging.info(f"Validation set size: {len(datasets['valid'])}")
        if test_files:
            logging.info(f"Test set size: {len(datasets['test'])}")

        dataloaders = {
            'train': DataLoader(datasets['train'], batch_size=batch_size[0], shuffle=False)
            if train_files is not None else None,
            'valid': DataLoader(datasets['valid'], batch_size=batch_size[1], shuffle=False)
            if valid_files is not None else None,
            'test': DataLoader(datasets['test'], batch_size=batch_size[2], shuffle=False)
            if test_files is not None else None
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
    def prepare_lines(data, split_on=None, cast_to=None):
        out = []
        for line in data:
            line = line.strip()
            if split_on:
                line = line.split(split_on)
                line = list(filter(None, line))
            else:
                line = [line]

            if cast_to:
                line = [cast_to(l) for l in line]

            out.append(line)

        return out

    def prepare_ms(self, data):
        seqs = []
        # Data is a 'text' of 'sentences'
        for line in data:
            # Every line is a 'sentence' of tab-separated 'tokens' 
            line = line.strip()
            # Every token is a 'word' which is a list of 0s and 1s
            tokens = line.split('\t')
            tokens = [list((map(int, token.split(' ')))) for token in tokens]
            seqs.append(np.array(tokens))

            # Get the length of the not-padded sequences
        lengths = torch.LongTensor([len(s) for s in seqs])

        # Create zero-only dataset                 
        seq_tensor = torch.zeros(len(seqs), lengths.max(), self.ms['dim'])

        # Fill in real values                 
        for idx, (seq, seqlen) in enumerate(zip(seqs, lengths)):
            seq_tensor[idx, :seqlen] = torch.FloatTensor(seq)

        # Gets back sorted lengths and the indices of sorted items
        lengths, sorted_idxs = torch.sort(lengths, dim=0, descending=True)
        # Sort tensor by using sorted indices
        seq_tensor = seq_tensor[sorted_idxs, :, :]
        # Pack sequences
        packed_seqs = pack_padded_sequence(seq_tensor, lengths, batch_first=True)

        return packed_seqs, sorted_idxs

    def prepare_w2v(self, data):
        """ Gets the word2vec ID of the tokens.
            Input is a batch of sentences, consisting of tokens. """
        def get_word_idx(word):
            try:
                return self.w2v['vocab'][word].index
            except KeyError:
                if self.w2v['unknown_token']:
                    try:
                        return self.w2v['vocab'][self.w2v['unknown_token']].index
                    except KeyError:
                        raise KeyError("The specified 'unknown_token' is not present in your word2vec model.")
                else:
                    raise KeyError('A word in your training set is not present in your word2vec model.')

        seqs = [np.array(list(map(get_word_idx, sent))) for sent in data]

        # Get the length of the not-padded sequences
        lengths = torch.LongTensor([len(s) for s in seqs])

        # fill tensor with index of padding
        seq_tensor = torch.full((len(seqs), lengths.max().item()), self.w2v['padding_idx'])

        # Fill in real values
        for idx, (seq, seqlen) in enumerate(zip(seqs, lengths)):
            seq_tensor[idx, :seqlen] = torch.FloatTensor(seq)

        return seq_tensor.long()

    def prepare_fasttext(self, data):
        """ Converts an input list of lists of tokens to packed sequences of fastText vectors. """
        # Data is a 'text' of 'sentences'
        seqs = [np.array(list(map(self.fasttext['model'].get_word_vector, sent))) for sent in data]

        # Get the length of the not-padded sequences
        lengths = torch.LongTensor([len(s) for s in seqs])

        # Create zero-only dataset
        seq_tensor = torch.zeros(len(seqs), lengths.max(), self.fasttext['dim'])

        # Fill in real values
        for idx, (seq, seqlen) in enumerate(zip(seqs, lengths)):
            seq_tensor[idx, :seqlen] = torch.FloatTensor(seq)

        # Gets back sorted lengths and the indices of sorted items
        lengths, sorted_idxs = torch.sort(lengths, dim=0, descending=True)
        # Sort tensor by using sorted indices
        seq_tensor = seq_tensor[sorted_idxs, :, :]
        # Pack sequences
        packed_seqs = pack_padded_sequence(seq_tensor, lengths, batch_first=True)

        return packed_seqs, sorted_idxs

    @staticmethod
    def prepare_elmo(sentences):
        # Add <S> and </S> tokens to sentence
        # See https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
        # #notes-on-statefulness-and-non-determinism
        elmo_sentences = []
        for s in sentences:
            elmo_sentences.append(['<S>', *s, '</S>'])

        return elmo_sentences

    def prepare_bert(self, sentences):
        all_input_ids = []
        all_input_mask = []
        for sentence in sentences:
            sentence = ' '.join(sentence)
            # tokenizer will also separate on punctuation
            # see https://github.com/google-research/bert#tokenization
            tokens = self.bert['tokenizer'].tokenize(sentence)

            # limit size of tokens (-2 to account for CLS and SEP
            if len(tokens) > self.bert['max_seq_len'] - 2:
                tokens = tokens[0:(self.bert['max_seq_len'] - 2)]

            # add [CLS] and [SEP], as expected in BERT
            tokens = ['[CLS]', *tokens, '[SEP]']

            input_ids = self.bert['tokenizer'].convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < self.bert['max_seq_len']:
                input_ids.append(0)
                input_mask.append(0)

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)

        all_input_ids = torch.LongTensor(all_input_ids)
        all_input_mask = torch.FloatTensor(all_input_mask)

        return all_input_ids, all_input_mask

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

            train_loss, train_results = self._process('train', log_update_freq, epoch)
            total_train_time += time.time() - epoch_start

            try:
                train_pearson = pearsonr(train_results['predictions'], train_results['targets'])
            except FloatingPointError:
                train_pearson = "Could not calculate Pearsonr"

            # Calculate average losses
            train_loss = np.mean(train_loss)
            train_losses.append(train_loss)

            # VALIDATION
            valid_loss, valid_results = self._process('valid', log_update_freq, epoch)

            try:
                valid_pearson = pearsonr(valid_results['predictions'], valid_results['targets'])
            except FloatingPointError:
                valid_pearson = "Could not calculate Pearsonr"

            valid_loss = np.mean(valid_loss)
            valid_losses.append(valid_loss)

            # Log epoch statistics
            logging.info(f"Epoch {epoch} - completed in {(time.time() - epoch_start):.0f} seconds"
                         f"\nTraining Loss: {train_loss:.6f}\t Pearson: {train_pearson}"
                         f"\nValidation loss: {valid_loss:.6f}\t Pearson: {valid_pearson}")

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

        results = {'predictions': np.array([]), 'targets': np.array([])}
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
            if self.ms['use']:
                ms = data['ms']
            if self.use_sentences:
                sentences = data['sentences']

            target = data['labels']

            sorted_ids = None
            # Convert ms features to int array
            if self.ms['use']:
                ms, sorted_ids = self.prepare_ms(ms)
                ms = ms.to(self.device)
            else:
                ms = None

            # Convert sentence to token array
            if self.use_sentences:
                sentences = self.prepare_lines(sentences, split_on=' ')

            if self.fasttext['use']:
                fasttext_vec, sorted_ids = self.prepare_fasttext(sentences)
                fasttext_vec = fasttext_vec.to(self.device)
            else:
                fasttext_vec = None

            # Convert tokens to word2vec IDs
            if self.w2v['use']:
                w2v_ids = self.prepare_w2v(sentences)
                w2v_ids = w2v_ids[sorted_ids] if sorted_ids is not None else w2v_ids
                w2v_ids = w2v_ids.to(self.device)
            else:
                w2v_ids = None

            if self.elmo['use']:
                elmo_sentence = self.prepare_elmo(sentences)
                elmo_ids = batch_to_ids(elmo_sentence)
                elmo_ids = elmo_ids[sorted_ids] if sorted_ids is not None else elmo_ids
                elmo_ids = elmo_ids.to(self.device)
            else:
                elmo_ids = None

            if self.bert['use']:
                bert_ids, bert_mask = self.prepare_bert(sentences)
                bert_ids = bert_ids[sorted_ids] if sorted_ids is not None else bert_ids
                bert_mask = bert_mask[sorted_ids] if sorted_ids is not None else bert_mask

                bert_ids = bert_ids.to(self.device)
                bert_mask = bert_mask.to(self.device)
                bert_input = (bert_ids, bert_mask)
            else:
                bert_input = None

            # Convert target to float array
            target = torch.FloatTensor(self.prepare_lines(target, cast_to=float))
            # Sort targets in-line with previous sorting
            target = target[sorted_ids] if sorted_ids is not None else target
            # Get current batch size
            curr_batch_size = target.size(0)

            target = target.to(self.device)

            # 2. Predictions
            pred = self.model(ms, w2v_ids, fasttext_vec, elmo_ids, bert_input)
            loss = self.criterion(pred, target)

            # 3. Optimise during training
            if do == 'train':
                loss.backward()
                self.optimizer.step()

            # 4. Save results
            pred = pred.detach().cpu().numpy()
            target = target.cpu().numpy()

            results['predictions'] = np.append(results['predictions'], pred, axis=None)
            results['targets'] = np.append(results['targets'], target, axis=None)
            losses = np.append(losses, float(loss))

            if log_update_freq and batch_idx in update_checkpoints:
                if do in ('train', 'valid'):
                    logging.info(f"{do.capitalize()} epoch {epoch}, batch nr. {batch_idx}/{nro_batches}...")
                else:
                    logging.info(f"{do.capitalize()}, batch nr. {batch_idx}/{nro_batches}...")

        return losses, results

    def test(self, checkpoint_f='checkpoint.pth', log_update_freq=0):
        logging.info('Testing started.')
        test_start = time.time()

        if self.checkpoint_f is None:
            self.model.load_state_dict(torch.load(checkpoint_f, map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(self.checkpoint_f, map_location=self.device))

        test_loss, test_results = self._process('test', log_update_freq)

        try:
            test_pearson = pearsonr(test_results['predictions'], test_results['targets'])
        except FloatingPointError:
            test_pearson = "Could not calculate Pearsonr"

        test_loss = np.mean(test_loss)

        logging.info(f"Testing completed in {(time.time() - test_start):.0f} seconds"
                     f"\nLoss: {test_loss:.6f}\t Pearson: {test_pearson}\n")

        return test_loss, test_pearson[0]