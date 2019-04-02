from torch.utils.data import Dataset

import linecache


class LazyTextDataset(Dataset):
    def __init__(self, paths):
        self.paths, self.labels_path = paths[:-1], paths[-1]

        with open(self.labels_path, encoding='utf-8') as fhin:
            lines = 0
            for line in fhin:
                if line.strip() != '':
                    lines += 1

            self.num_entries = lines

    def __getitem__(self, idx):
        data = [linecache.getline(p, idx + 1) for p in self.paths]
        label = linecache.getline(self.labels_path, idx + 1)

        return (*data, label)

    def __len__(self):
        return self.num_entries
