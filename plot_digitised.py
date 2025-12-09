# plot_digitized.py
import numpy as np, matplotlib.pyplot as plt
a = np.load("digitized_for_model/Screenshot 2025-12-11 201727.npy")
print("shape:", a.shape, "min/max:", a.min(), a.max())
t = np.arange(a.shape[1]) / 500.0  # seconds (if 500Hz)
plt.figure(figsize=(12, 10))
for i in range(12):
    plt.subplot(6,2,i+1)
    plt.plot(t, a[i])
    plt.title(f"Lead {i+1}")
    plt.ylabel("mV")
    plt.xlim(0, t[-1])
plt.tight_layout()
plt.show()
