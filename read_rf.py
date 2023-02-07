import matlab.engine
import numpy as np
from numpy import savez_compressed

eng = matlab.engine.start_matlab()


def read(filepath):
    rf = eng.RPread(filepath)
    rf_np = np.array(rf._data)
    rf_np = np.reshape(rf_np, rf.size, order="F")
    return rf_np


def save_rf(filepath_to_be_read, filepath_to_be_saved):
    rf = eng.RPread(filepath_to_be_read)
    rf_np = np.array(rf._data)
    rf_np = np.reshape(rf_np, rf.size, order="F")
    print(rf_np.shape)
    savez_compressed(filepath_to_be_saved, rf_np)
    return None


