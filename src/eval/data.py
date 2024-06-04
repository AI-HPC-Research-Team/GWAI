"""data evaluation and visualization functions"""

import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller

from .utils import Constant


def tukey(M, alpha=0.5):
    """
    Tukey window code copied from scipy

    Args:
        M (int): The window length.
        alpha (float, optional): The alpha parameter. Defaults to 0.5.
    """
    n = np.arange(0, M)
    width = int(np.floor(alpha * (M - 1) / 2.0))
    n1 = n[0 : width + 1]
    n2 = n[width + 1 : M - width - 1]
    n3 = n[M - width - 1 :]

    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0 * n1 / alpha / (M - 1))))
    w2 = np.ones(n2.shape)
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (M - 1))))
    w = np.concatenate((w1, w2, w3))

    return np.array(w[:M])


def get_amp_snr(sig, noise):
    """Get the signal-to-noise ratio (SNR) of a signal given the noise.

    Args:
        sig (array): The signal.
        noise (array): The noise.

    Returns:
        float: The SNR.
    """
    return np.sqrt(np.sum(sig**2) / np.sum(noise**2))


def get_mf_snr(
    data,
    T_obs,
    fs,
    fmin,
    psd,
):
    """
    computes the snr of a signal given a PSD starting from a particular frequency index

    Args:
        data (array): The data.
        T_obs (float): The observation time.
        fs (float): The sampling frequency.
        fmin (float): The minimum frequency.
        psd (array): The psd.
    """
    N = int(T_obs * fs)
    df = 1.0 / T_obs
    dt = 1.0 / fs
    fidx = int(fmin / df)

    win = tukey(N, alpha=1.0 / 8.0)
    idx = np.argwhere(psd > 0.0)
    invpsd = np.zeros(psd.size)
    invpsd[idx] = 1.0 / psd[idx]

    xf = np.fft.rfft(data * win) * dt

    SNRsq = 4.0 * np.sum((np.abs(xf[fidx:]) ** 2) * invpsd[fidx:]) * df
    return np.sqrt(SNRsq)


def ADF_test(x):
    """
    Augmented Dickey-Fuller test for stationarity

    Args:
        x (array): The data.
    """
    result = adfuller(x)
    return result[1]


def shapiro_wilks_test(x):
    """
    Shapiro-Wilks test for normality

    Args:   
        x (array): The data.
    """
    return shapiro(x)


def plot_td_data(t, data):
    """
    Plot time domain data

    Args:
        t (array): The time.
        data (array): The data.
    """
    plt.plot(t / Constant.YRSID_SI, data)
    plt.xlabel("Time [yr]")
    plt.ylabel("Strain")
    plt.show()


def plot_fd_data(f, data, psd):
    """
    Plot frequency domain data psd

    Args:
        f (array): The frequency.
        data (array): The data.
        psd (array): The psd.
    """
    f, psd_data = signal.welch(data, fs=1.0 / (f[1] - f[0]), nperseg=512 * 512)
    plt.loglog(f, psd_data, label="Data")
    plt.loglog(f, psd, label="PSD")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Strain")
    plt.show()


def plot_spectrogram(data, fs):
    """
    Plot spectrogram

    Args:
        data (array): The data.
        fs (float): The sampling frequency.
    """
    f, t, Sxx = signal.spectrogram(data, fs, nperseg=512, noverlap=256)
    plt.pcolormesh(t / Constant.YRSID_SI, f, np.log10(Sxx))
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [yr]")
    plt.show()
