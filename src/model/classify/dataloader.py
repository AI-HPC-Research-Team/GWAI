# mypy: ignore-errors
# type: ignore
import logging

import numpy as np
import torch

# import h5py
from torch.utils.data import Dataset

from pathlib import Path

import h5py

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

# type: ignore


# mylogger = MyLogger(__name__)


def save_waveform(data=None, DIR=".", data_fn="waveform_dataset.hdf5"):
    """Save waveform dataset to hdf5 file.

    Parameters
    ----------
    data : list, optional
        Dataset to save, [waveform_dataset, waveform_params]
    DIR : str, optional
        Specified directory to save waveform dataset, by default '.'
    data_fn : str, optional
        Specified file name to save waveform dataset, by default 'waveform_dataset.hdf5'
    """
    if data is None:
        print("No data to save!")
        return
    wfd, wfp = data
    p = Path(DIR)
    p.mkdir(parents=True, exist_ok=True)
    # mylogger.logger.info("Saving waveforms...")
    f_data = h5py.File(p / data_fn, "w")

    data_name = "0"
    for i in wfd.keys():
        for j in wfd[i].keys():
            data_name = i + "_" + j
            f_data.create_dataset(
                data_name,
                data=wfd[i][j],
                compression="gzip",
                compression_opts=9,
            )

    for i in wfp.keys():
        data_name = i + "_" "par"
        f_data.create_dataset(
            data_name, data=wfp[i], compression="gzip", compression_opts=9
        )
    f_data.close()


def load_waveform(data=None, DIR=".", data_fn="waveform_dataset.hdf5"):
    """load waveform dataset from hdf5 file.

    Parameters
    ----------
    DIR : str, optional
        Specified directory to the waveform dataset, by default '.'
    data_fn : str, optional
        Specified file name to the waveform dataset, by default 'waveform_dataset.hdf5'
    """
    # Load data from HDF5 file
    if data is None:
        print("No data to save!")
        return
    wfd, wfp = data
    p = Path(DIR)
    f_data = h5py.File(p / data_fn, "r")
    # mylogger.logger.info("Loading waveforms...")
    data_name = "0"
    # Load parameters
    for i in wfp.keys():
        data_name = i + "_" "par"
        try:
            wfp[i] = f_data[data_name][()]
        except KeyError:
            print("Could not find dataset with name %s" % data_name)
    # Load waveform data
    for i in wfd.keys():
        for j in wfd[i].keys():
            data_name = i + "_" + j
            try:
                wfd[i][j] = f_data[data_name][()]
            except KeyError:
                print("Could not find dataset with name %s" % data_name)
    f_data.close()
