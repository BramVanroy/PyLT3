import h5py
import torch
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(self, paths):
        paths, labels_path = paths[:-1], paths[-1]
        self.labels = h5py.File(str(labels_path)).get('utils')
        self.data = [h5py.File(str(p)).get('utils') for p in paths]
        self.num_entries = self.labels.shape[0]

    def __getitem__(self, index):
        data = [torch.from_numpy(d[index, :]).float() for d in self.data]
        label = torch.from_numpy(self.labels[index, :]).float()

        return (*data, label)

    def __len__(self):
        return self.num_entries
