# preprocess.py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample

FS = 500
DURATION = 10
TARGET_LEN = FS * DURATION
N_LEADS = 12

def bandpass_filter(x, fs=FS, low=0.5, high=40.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def notch_filter(x, fs=FS, freq=50.0, q=30.0):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, x)

def pad_or_crop(sig, target_len=TARGET_LEN):
    T = sig.shape[1]
    if T > target_len:
        return sig[:, :target_len]
    elif T < target_len:
        pad = target_len - T
        return np.pad(sig, ((0,0),(0,pad)), mode='edge')
    return sig

def resample_to_target(sig_12xT, target_len=TARGET_LEN):
    out = np.zeros((N_LEADS, target_len), dtype=np.float32)
    for i in range(N_LEADS):
        out[i] = resample(sig_12xT[i], target_len)
    return out

def preprocess_ecg(signal_Tx12):
    """
    Input: signal_Tx12 shape [T,12]
    Output: np.ndarray [12, TARGET_LEN]
    """
    sig = signal_Tx12.T.astype(np.float32)   # [12, T]
    sig = pad_or_crop(sig, TARGET_LEN)
    for i in range(N_LEADS):
        lead = sig[i]
        lead = bandpass_filter(lead)
        lead = notch_filter(lead)
        lead = (lead - lead.mean()) / (lead.std() + 1e-7)
        sig[i] = lead
    return sig
