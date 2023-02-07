import torch
import torch.nn as nn
from scipy import signal
import matlab.engine
import numpy as np


eng = matlab.engine.start_matlab()


class Denoising(nn.Module):

    def __init__(self, low=6, high=-60, fs=4e7):
        super(Denoising, self).__init__()

        samplingFrequency = fs

        tpCount = 256
        values = np.arange(int(tpCount / 2))
        timePeriod = tpCount / samplingFrequency
        frequencies = values / timePeriod

        desired_filter = np.zeros(tpCount // 2 + 1)
        desired_filter[low:high] = 1

        bands = np.append(frequencies, frequencies[-1] + frequencies[1])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kernel = signal.firwin2(51, bands, desired_filter, fs=fs)[:, np.newaxis, np.newaxis]

    def forward(self, x):
        x = signal.convolve(x, self.kernel, mode='same')
        return x
