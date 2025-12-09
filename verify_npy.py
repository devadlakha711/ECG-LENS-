# verify_npy.py
import numpy as np, os, sys

# Edit these paths if needed:
orig_dir = "digitized_signals"         # original outputs
mV_dir   = "digitized_signals_mV"      # your converted outputs
fname = "img1.npy"

p_orig = os.path.join(orig_dir, fname)
p_mV   = os.path.join(mV_dir, fname)

for p,label in [(p_orig,"ORIG"), (p_mV,"mV")]:
    if os.path.exists(p):
        a = np.load(p)
        print(f"{label}: {p} exists. shape={a.shape}, min={a.min():.6g}, max={a.max():.6g}, mean={a.mean():.6g}")
    else:
        print(f"{label}: {p} MISSING")
