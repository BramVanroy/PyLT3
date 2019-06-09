from torch import nn


class CharacterRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0, bidirectional=True):
        super(CharacterRNN, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rlayer = nn.GRU(input_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.dropout = dropout
        self.droplayer = nn.Dropout(dropout)
        self.llayer = nn.Linear(hidden_size, output_size)

    def forward(self, batch_input):
        r_out, _ = self.rlayer(batch_input)

        if self.bidirectional:
            r_out = r_out[:, :, :self.hidden_size] + r_out[:, :, self.hidden_size:]

        last_r_out = r_out[:, -1, :]
        drop_out = self.droplayer(last_r_out) if self.dropout > 0 else last_r_out
        l_out = self.llayer(drop_out)

        return l_out
