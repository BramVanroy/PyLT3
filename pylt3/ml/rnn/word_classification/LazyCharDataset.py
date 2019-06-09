import linecache
from pathlib import Path

from torch.utils.data import Dataset


class LazyCharDataset(Dataset):
    def __init__(self, fname):
        self.fname = str(Path(fname).resolve())

        with open(self.fname, encoding='utf-8') as fhin:
            lines = 0
            for line in fhin:
                if line.strip() != '':
                    lines += 1

            self.num_entries = lines

    def __getitem__(self, idx):
        return linecache.getline(self.fname, idx + 1).strip().split('\t')

    def __len__(self):
        return self.num_entries
