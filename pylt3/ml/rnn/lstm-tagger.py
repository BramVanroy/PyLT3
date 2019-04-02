from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gensim

torch.manual_seed(1)


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
            self.hidden_w2v = self.init_hidden()

        if ms_dim is not None:
            self.ms_lstm = nn.LSTM(ms_dim, hidden_dim, bidirectional=True)
            self.hidden_ms = self.init_hidden()

        # The linear layer that maps from hidden state space to a single output
        self.linear = nn.Linear(1, 1)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))

    def forward(self, batch_size, sentence_input=None, ms_input=None):
        if sentence_input is not None:
            embeds = self.word_embeddings(sentence_input)
            w2v_out, self.hidden_w2v = self.w2v_lstm(embeds.view(len(sentence_input), batch_size, -1), self.hidden_w2v)
            w2v_first_bi = w2v_out[:, :, :self.hidden_dim]
            w2v_last_bi = w2v_out[:, :, self.hidden_dim:]
            w2v_sum_bi = (w2v_first_bi + w2v_last_bi)

        if ms_input is not None:
            ms_out, self.hidden_ms = self.ms_lstm(ms_input.view(-1, batch_size, len(ms_input)), self.hidden_ms)
            ms_first_bi = ms_out[:, :, :self.hidden_dim]
            ms_last_bi = ms_out[:, :, self.hidden_dim:]
            ms_sum_bi = ms_first_bi + ms_last_bi

        if not (sentence_input is None or ms_input is None):
            summed = w2v_sum_bi + ms_sum_bi
        elif ms_input is None:
            summed = w2v_sum_bi
        else:
            summed = ms_sum_bi

        print(summed.shape)
        regression = F.relu(self.linear(summed.view(-1, 1, 1)[-1])[-1])

        return regression


def prepare_sequence(seq, weights):
    idxs = [weights.vocab[w].index for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# ================================================

training_data = [
    ("the dog ate the apple".split(), [0, 0, 0, 1], 0.25),
    ("everybody read that book".split(), [0, 1, 1, 0], 13.452)
]

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# ================================================

# Init model
embed_p = Path('..\..\..\data\dpc\ml\other\dpc+news2017.dim146-ep10-min2-win10-repl.w2v_model').resolve()
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(str(embed_p))
embed_weights = torch.FloatTensor(w2v_model.vectors)

HIDDEN_DIM = 200
MS_SIZE = 4
BATCH_SIZE = 1

model = LSTMRegressor(HIDDEN_DIM, MS_SIZE, embed_weights)
# model = LSTMRegressor(HIDDEN_DIM, MS_SIZE)
loss_function = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

# See what the results are before training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], w2v_model)
    ms_in = torch.tensor(training_data[0][1], dtype=torch.float)
    regr = model(BATCH_SIZE, sentence_input=inputs, ms_input=ms_in)
    print(regr)

    inputs = prepare_sequence(training_data[1][0], w2v_model)
    ms_in = torch.tensor(training_data[1][1], dtype=torch.float)
    regr = model(BATCH_SIZE, sentence_input=inputs, ms_input=ms_in)
    print(regr)

for epoch in range(20):
    for sentence, ms, target in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden_w2v = model.init_hidden()
        model.hidden_ms = model.init_hidden()
        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, w2v_model)
        ms_in = torch.tensor(ms, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)

        # Step 3. Run our forward pass.
        score = model(BATCH_SIZE, sentence_input=None, ms_input=ms_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(score, target)
        # must change this to minimise memory footprint
        loss.backward(retain_graph=True)
        optimizer.step()

# See what the results are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], w2v_model)
    ms_in = torch.tensor(training_data[0][1], dtype=torch.float)
    regr = model(BATCH_SIZE, sentence_input=inputs, ms_input=ms_in)
    print(regr)

    inputs = prepare_sequence(training_data[1][0], w2v_model)
    ms_in = torch.tensor(training_data[1][1], dtype=torch.float)
    regr = model(BATCH_SIZE, sentence_input=inputs, ms_input=ms_in)
    print(regr)
