import logging
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(r'..\..\..\PyLT3')
from pylt3.ml.rnn.LazyTextDataset import LazyTextDataset

# Make results reproducible
torch.manual_seed(3)
torch.backends.cudnn.deterministic = True
np.random.seed(3)

# run all numpy warnings as errors

np.seterr(all='raise')

