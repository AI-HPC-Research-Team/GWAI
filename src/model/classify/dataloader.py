# mypy: ignore-errors
# type: ignore
import logging

import numpy as np
import torch

# import h5py
from torch.utils.data import Dataset

from ...utils.io.hdf5_wfd import load_waveform, save_waveform

log = logging.getLogger(__name__)


class TinyEMRIDataset(object):
    def __init__(self, DIR, fn):
        """
        Load the TinyEMRI Dataset

        Args:
            DIR (str): Directory of the data
            fn (str): Filename of the data
        """
        super().__init__()
        # self.train = train
        self.data = {
            "train": {"signal": [], "noise": []},
            "test": {"signal": [], "noise": []},
        }
        self.label = {"train": [], "test": []}
        self.params = {"train": [], "test": []}

        log.info("Loading data from {}/{}".format(DIR, fn))
        load_waveform(data=[self.data, self.params], DIR=DIR, data_fn=fn)

    def save(self, DIR, fn):
        save_waveform(data=[self.data, self.params], DIR=DIR, data_fn=fn)


class EMRIDatasetTorch(Dataset):
    def __init__(self, wfd, train=True):
        """
        Load the EMRI Dataset

        Args:
            wfd (TinyEMRIDataset): EMRI Dataset
            train (bool, optional): Train or test. Defaults to True.
        """
        super().__init__()
        self.wfd = wfd
        self.train = train
        self.type_str = "train" if self.train else "test"
        self.length = self.wfd.data[self.type_str]["signal"].shape[-1]
        # self.n_channel = self.wfd.data[self.type_str][
        #     "signal"
        # ].shape[-2]
        self.n_signal = self.wfd.data[self.type_str]["signal"].shape[0]

    def __len__(self):
        return (
            self.wfd.data[self.type_str]["noise"].shape[0]
            + self.wfd.data[self.type_str]["signal"].shape[0]
        )
        # for activation map
        # return 2

    def __getitem__(self, idx):
        if idx < self.n_signal:
            # for activation map
            # if idx == 0:
            data = (
                self.wfd.data[self.type_str]["signal"][idx][:, ::4]
                # + self.wfd.data[self.type_str]["noise"][idx]
            )
            label = 1
        else:
            # idx -= self.n_signal
            data = self.wfd.data[self.type_str]["noise"][idx - self.n_signal][:, ::4]
            label = 0
        # data = np.log10(np.abs(data) + 1e-20)
        if np.isnan(data).any():
            raise ValueError("NaN in data")

        return (
            torch.tensor(idx, dtype=torch.long),
            torch.from_numpy(data).float(),
            torch.tensor(label, dtype=torch.long),  # for nn.CrossEntropyLoss
            # torch.tensor(label, dtype=torch.float),  # for nn.BCEWithLogitsLoss
        )


class NpzDatasetTorch(Dataset):
    def __init__(self, fn, train=False):
        """
        Load the EMRI Dataset from .npz file

        Args:
            fn (str): Filename of the data
            train (bool, optional): Train or test. Defaults to False.
        """
        super().__init__()
        self.data = np.load(fn)
        # check the dimension of the data
        if len(self.data.shape) == 2:
            self.data = self.data[np.newaxis, :]

    def __len__(self):
        return self.data.shape[0]
        # for activation map
        # return 2

    def __getitem__(self, idx):
        data = self.data[idx]
        label = 1

        if np.isnan(data).any():
            raise ValueError("NaN in data")

        return (
            torch.tensor(idx, dtype=torch.long),
            torch.from_numpy(data).float(),
            torch.tensor(label, dtype=torch.long),  # for nn.CrossEntropyLoss
            # torch.tensor(label, dtype=torch.float),  # for nn.BCEWithLogitsLoss
        )
