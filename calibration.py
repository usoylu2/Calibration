import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import random
import matlab.engine
import numpy as np
from denoising import Denoising
from read_rf import read

Noise_low = 6
Noise_high = -60
denoising_func = Denoising(low=Noise_low, high=Noise_high)
eng = matlab.engine.start_matlab()
start_depth = 540
patch_size = 200
jump = 100
Depth = 9


class test_time_calibration(nn.Module):
    def __init__(self, Freq=False, Focus=False, Power=False, Noise=False, device="cuda", fs=4e7, filter_length=51):
        super(test_time_calibration, self).__init__()

        if Freq:
            fileNames = ['//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/stable/glassbeads/ufuk1.rf',
                '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/stable/glassbeads/ufuk9.rf']
        elif Focus:
            fileNames = ['//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/stable/focus_specific/glass9MHz2cm.rf',
                        '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/stable/focus_specific/glass9MHz1cm3cm.rf']
        elif Power:
            fileNames = ['//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/stable/power_specific/glass9MHz0db.rf',
                        '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/stable/power_specific/glass9MHz6db.rf']
        else:
            print("Error in mismatch type")

        samplingFrequency = 40000000
        tpCount = patch_size
        values = np.arange(int(tpCount / 2))
        timePeriod = tpCount / samplingFrequency
        frequencies = values / timePeriod
        sqfourierTransform = np.zeros((Depth, int(len(fileNames)), int(tpCount / 2)))

        for h in range(len(fileNames)):
            print(fileNames[h])
            rf_np = read(fileNames[h])
            if Noise:
                rf_np = denoising_func(rf_np)
            for depth_index in range(Depth):
                for j in range(rf_np.shape[2]):
                    for i in range(rf_np.shape[1]):
                        amplitude = rf_np[start_depth + depth_index * jump: start_depth +
                                                                            patch_size + depth_index * jump, i, j]
                        # Frequency domain representation
                        fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
                        fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
                        sqfourierTransform[depth_index, h, :] = sqfourierTransform[depth_index, h, :] \
                                                                + pow(abs(fourierTransform), 2)
                sqfourierTransform[depth_index, h] = sqfourierTransform[depth_index, h] / \
                                                     (rf_np.shape[2] * rf_np.shape[1])

        noise_pow = np.min(sqfourierTransform, axis=2)
        snr = ((sqfourierTransform-noise_pow[:, :, np.newaxis])+1e-20)/noise_pow[:, :, np.newaxis]
        snr = np.append(snr, snr[:, :, -1][:, :, np.newaxis], axis=2)
        snr = np.min(snr, axis=1)

        h_abs = np.zeros((Depth, tpCount // 2 + 1))
        for depth_index in range(Depth):
            temp = np.sqrt(sqfourierTransform[depth_index, 1] /
                           (sqfourierTransform[depth_index, 0] + 1e-20))
            h_abs[depth_index] = np.append(temp, temp[-1])
        h_abs = np.abs(h_abs)
        wiener_res = h_abs / (h_abs * h_abs + 1 / snr)

        self.desired = wiener_res
        self.bands = np.append(frequencies, frequencies[-1] + frequencies[1])
        self.fs = fs
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kernel = []

        for depth_index in range(Depth):
            temp = torch.from_numpy(signal.firwin2(filter_length, self.bands, self.desired[depth_index],
                                                   fs=self.fs)[np.newaxis, np.newaxis, :, np.newaxis])
            self.kernel.append(temp.type(torch.FloatTensor).to(device))

    def forward(self, x, y, depth):
        for i in range(x.shape[0]):
            x[i, :, :, :] = F.conv2d(x[i][np.newaxis, :, :, :], self.kernel[depth[i]], padding='same')
        return x


class train_time_calibration(nn.Module):
    def __init__(self, Freq=False, Focus=False, Power=False, Noise=False, device="cuda", fs=4e7,
                 probability=0.5, filter_length=51):
        super(train_time_calibration, self).__init__()

        if Freq:
            fileNames = ['//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/stable/glassbeads/ufuk1.rf',
                '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/stable/glassbeads/ufuk9.rf']
        elif Focus:
            fileNames = ['//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/stable/focus_specific/glass9MHz2cm.rf',
                        '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/stable/focus_specific/glass9MHz1cm3cm.rf']
        elif Power:
            fileNames = ['//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/stable/power_specific/glass9MHz0db.rf',
                        '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/stable/power_specific/glass9MHz6db.rf']
        else:
            print("Error in mismatch type")

        samplingFrequency = 40000000
        tpCount = patch_size
        values = np.arange(int(tpCount / 2))
        timePeriod = tpCount / samplingFrequency
        frequencies = values / timePeriod
        sqfourierTransform = np.zeros((Depth, int(len(fileNames)), int(tpCount / 2)))

        for h in range(len(fileNames)):
            print(fileNames[h])
            rf_np = read(fileNames[h])
            if Noise:
                 rf_np = denoising_func(rf_np)
            for depth_index in range(Depth):
                for j in range(rf_np.shape[2]):
                    for i in range(rf_np.shape[1]):
                        amplitude = rf_np[start_depth + depth_index * jump: start_depth + patch_size + depth_index * jump, i, j]
                        # Frequency domain representation
                        fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
                        fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
                        sqfourierTransform[depth_index, h, :] = sqfourierTransform[depth_index, h, :] \
                                                                + pow(abs(fourierTransform), 2)
                sqfourierTransform[depth_index, h] = sqfourierTransform[depth_index, h] / \
                                                     (rf_np.shape[2] * rf_np.shape[1])

        noise_pow = np.min(sqfourierTransform, axis=2)
        snr = ((sqfourierTransform-noise_pow[:, :, np.newaxis])+1e-20)/noise_pow[:, :, np.newaxis]
        snr = np.append(snr, snr[:, :, -1][:, :, np.newaxis], axis=2)
        snr = np.min(snr, axis=1)

        h_abs = np.zeros((Depth, tpCount // 2 + 1))
        for depth_index in range(Depth):
            temp = np.sqrt(sqfourierTransform[depth_index, 0] /
                           (sqfourierTransform[depth_index, 1] + 1e-20))
            h_abs[depth_index] = np.append(temp, temp[-1])
        h_abs = np.abs(h_abs)
        wiener_res = h_abs / (h_abs * h_abs + 1 / snr)
        self.p = probability
        self.desired = wiener_res
        self.bands = np.append(frequencies, frequencies[-1] + frequencies[1])
        self.fs = fs
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kernel = []

        for depth_index in range(Depth):
            temp = torch.from_numpy(signal.firwin2(filter_length, self.bands, self.desired[depth_index],
                                                   fs=self.fs)[np.newaxis, np.newaxis, :, np.newaxis])
            self.kernel.append(temp.type(torch.FloatTensor).to(device))

    def forward(self, x, y, depth):
        if random.uniform(0, 1) < self.p:
            for i in range(x.shape[0]):
                x[i, :, :, :] = F.conv2d(x[i][np.newaxis, :, :, :], self.kernel[depth[i]], padding="same")
            return x
        else:
            return x