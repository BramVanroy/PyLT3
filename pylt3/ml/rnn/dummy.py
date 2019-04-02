import torch
import torch.nn as nn
import torch.nn.functional as F

LSTM_DIMS = 12
BATCH_SIZE = 2
input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 20, 30, 40, 50, 60, 70, 80, 90]], dtype=torch.float)

INPUT_SIZE = input.size(1)

input = input.view(-1, BATCH_SIZE, INPUT_SIZE)
print(input.shape)
lstm = nn.LSTM(INPUT_SIZE, LSTM_DIMS, bidirectional=True)
hc = (torch.zeros(2, BATCH_SIZE, LSTM_DIMS), torch.zeros(2, BATCH_SIZE, LSTM_DIMS))
out, h = lstm(input, hc)
print(out.shape)

last_out = out[:, :, LSTM_DIMS:]
first_out = out[:, :, :LSTM_DIMS]

print(last_out)
print(first_out)

sum_out = first_out + last_out
print(sum_out)
fc = nn.Linear(LSTM_DIMS, 1)

fc_out = F.relu(fc(sum_out))

print(fc_out)


