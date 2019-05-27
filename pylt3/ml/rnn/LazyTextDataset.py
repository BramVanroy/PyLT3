import linecache
from pathlib import Path

from torch.utils.data import Dataset


class LazyTextDataset(Dataset):
    def __init__(self, paths_obj):
        self.ms_p = str(Path(paths_obj['ms']).resolve()) if 'ms' in paths_obj else None
        self.sentences_p = str(Path(paths_obj['sentences']).resolve()) if 'sentences' in paths_obj else None
        self.labels_p = str(Path(paths_obj['labels']).resolve())

        with open(self.labels_p, encoding='utf-8') as fhin:
            lines = 0
            for line in fhin:
                if line.strip() != '':
                    lines += 1

            self.num_entries = lines

    def __getitem__(self, idx):
        d = {
            'ms': linecache.getline(self.ms_p, idx + 1) if self.ms_p else None,
            'sentences': linecache.getline(self.sentences_p, idx + 1) if self.sentences_p else None,
            'labels': linecache.getline(self.labels_p, idx + 1)
        }

        return d

    def __len__(self):
        return self.num_entries
