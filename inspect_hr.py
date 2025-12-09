# inspect_and_hr.py
import numpy as np, matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks

def bp_filter_arr(arr, fs=500.0, low=0.5, high=40.0, order=3):
    from scipy.signal import butter, filtfilt
    nyq = 0.5*fs
    b,a = butter(order, [low/nyq, high/nyq], btype='band')
    out = np.zeros_like(arr)
    for ii in range(arr.shape[0]):
        try:
            out[ii] = filtfilt(b, a, arr[ii], method='pad')
        except Exception:
            out[ii] = arr[ii]
    return out

def quick_hr_estimate(lead_signal, fs=500):
    import numpy as np
    from scipy.signal import find_peaks
    sig = lead_signal.copy()
    if np.isnan(sig).all():
        return None, []
    amp = np.nanmax(sig) - np.nanmin(sig)
    if amp <= 0 or np.isnan(amp): return None, []
    height = max(0.05 * amp, 0.02 * amp)
    prominence = max(0.05 * amp, 0.01)
    min_dist = int(fs * 0.25)
    peaks, props = find_peaks(np.abs(sig), distance=min_dist, height=height, prominence=prominence)
    if len(peaks) < 2:
        peaks, props = find_peaks(np.abs(sig), distance=int(fs*0.2), prominence=prominence*0.5)
    if len(peaks) < 2:
        return None, peaks
    rr = np.diff(peaks) / fs
    hr = 60.0 / np.mean(rr)
    return hr, peaks

sig = np.load("digitized_signals/img1.npy")
print("loaded shape:", sig.shape, "min/max:", np.nanmin(sig), np.nanmax(sig))
# clean step should already have been applied; bandpass now
sig_bp = bp_filter_arr(sig, fs=500)

vars = np.nanvar(sig_bp, axis=1)
for i in range(sig_bp.shape[0]):
    print(f"lead {i+1:02d}: var={vars[i]:.6g}, min={np.nanmin(sig_bp[i]):.6g}, max={np.nanmax(sig_bp[i]):.6g}")

best = int(np.nanargmax(vars))
print("Selected lead", best+1, "for HR")
hr, peaks = quick_hr_estimate(sig_bp[best])
print("HR:", hr, "peaks:", len(peaks))
# plot
t = np.arange(sig_bp.shape[1]) / 500.0
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(t, sig_bp[best])
plt.plot(peaks/500.0, sig_bp[best][peaks], 'r.')
plt.title(f"Lead {best+1} (HR={hr})")
plt.show()
