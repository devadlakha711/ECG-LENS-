# digitizer_runner.py (project root)
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.io import read_image, write_png, write_jpeg
from torchvision.transforms.functional import adjust_contrast, rgb_to_grayscale, adjust_gamma, adjust_sharpness
from ecg_digitiser.run_digitizer import digitize_image_from_path

INPUT_DIR = "ecg_images"
OUTPUT_DIR = "digitized_signals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE = os.path.join("ecg_digitiser", "weights")
SEG_WEIGHTS = os.path.join(BASE, "unet_weights_07072025.pt")
LAYOUT_WEIGHTS = os.path.join(BASE, "lead_name_unet_weights_07072025.pt")

RESAMPLE_SIZE = 3000 
TARGET_NUM_SAMPLES = 5000

def preprocess_enhance_only(path):
    """
    Applies enhancements (Gamma, Contrast, Sharpness) to help model detect faint lines.
    Does NOT crop the image.
    """
    img = read_image(path)
    if img.shape[0] == 4: img = img[:3, :, :]
    
    if img.shape[0] == 3: img = rgb_to_grayscale(img)
    img = img.float() / 255.0

    # 1. GAMMA: Darkens faint lines
    img = adjust_gamma(img, gamma=3.5) 

    # 2. CONTRAST: Separates signal from grid
    img = adjust_contrast(img, contrast_factor=2.0)

    # 3. SHARPNESS: Helps model define edges
    img = adjust_sharpness(img, sharpness_factor=2.0)

    # 4. Thicken Lines
    img_inv = 1.0 - img
    img_thick_inv = F.max_pool2d(img_inv, kernel_size=2, stride=1, padding=1)
    
    if img_thick_inv.shape[-1] > img.shape[-1]:
        img_thick_inv = img_thick_inv[..., :img.shape[-2], :img.shape[-1]]
    
    img_thick = 1.0 - img_thick_inv
    img_thick = (img_thick * 255).clamp(0, 255).byte()
    final_img = img_thick.repeat(3, 1, 1)
    
    temp_path = path.replace(".", "_clean.")
    if path.lower().endswith(".jpg") or path.lower().endswith(".jpeg"):
        write_jpeg(final_img, temp_path, quality=95)
    else:
        write_png(final_img, temp_path)
    
    return temp_path

def process_final_signal(sig):
    """
    Cleans up the signal (Interpolation + Scaling).
    """
    sig = sig * 0.001
    for i in range(sig.shape[0]):
        s = pd.Series(sig[i])
        s = s.interpolate(method='linear', limit_direction='both')
        s = s.fillna(0)
        sig[i] = s.to_numpy()
    return sig

def save_raw_rows(sig, base_name):
    """
    Saves each detected row as a separate file.
    Does NOT attempt to slice them into 12 leads.
    """
    folder_path = os.path.join(OUTPUT_DIR, base_name)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    
    num_rows = sig.shape[0]
    print(f"  > Saving {num_rows} separate waveforms to {folder_path}")

    for i in range(num_rows):
        # Naming convention: Row_1, Row_2, etc.
        # If it's the 4th row, it's likely the Rhythm strip.
        if i == 3:
            row_name = f"Row_{i+1}_Rhythm"
        else:
            row_name = f"Row_{i+1}"
            
        filename = f"{row_name}" 
        
        # Save NPY
        np.save(os.path.join(folder_path, filename + ".npy"), sig[i].astype(np.float32))
        
        # Save PNG
        plt.figure(figsize=(15, 3)) # Wider plot for full row
        plt.plot(sig[i])
        plt.title(f"{base_name} - {row_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, filename + ".png"))
        plt.close()

def save_stacked_plot(sig, base_name):
    """
    Saves all detected rows in one stacked image for easy verification.
    """
    num_rows = sig.shape[0]
    full_png_path = os.path.join(OUTPUT_DIR, base_name + "_stacked_rows.png")
    
    fig, axes = plt.subplots(num_rows, 1, figsize=(15, 4 * num_rows), sharex=True)
    
    # Handle case if only 1 row detected (axes is not iterable)
    if num_rows == 1:
        axes = [axes]

    for i in range(num_rows):
        ax = axes[i]
        label = f"Row {i+1}"
        if i == 3: label += " (Rhythm)"
        
        ax.plot(sig[i], color='k', linewidth=1)
        ax.set_ylabel(label, rotation=0, labelpad=40, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.suptitle(f"Detected Waveforms (Rows): {base_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(full_png_path)
    plt.close()

def run_all():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files: return

    print(f"Found {len(files)} images. Extracting raw rows...")

    for fname in files:
        ipath = os.path.join(INPUT_DIR, fname)
        base_name = os.path.splitext(fname)[0]
        temp_path = None

        try:
            # 1. Preprocess (Enhance ONLY)
            temp_path = preprocess_enhance_only(ipath)
            print(f"Processing {fname}...")
            
            # 2. Digitize
            raw_sig = digitize_image_from_path(
                SEG_WEIGHTS, LAYOUT_WEIGHTS, 
                temp_path,
                resample_size=RESAMPLE_SIZE, 
                target_num_samples=TARGET_NUM_SAMPLES,
                device="cpu"
            )
            
            # Ensure Shape is [Rows, Samples]
            # Usually we expect samples (5000) > rows (4 or 12)
            if raw_sig.shape[0] > raw_sig.shape[1]: 
                raw_sig = raw_sig.T
            
            print(f"  > Raw Signal Shape: {raw_sig.shape}")

            # 3. Clean up (Interpolate NaNs)
            sig = process_final_signal(raw_sig)
            
            # 4. Save Raw Rows
            save_raw_rows(sig, base_name)
            save_stacked_plot(sig, base_name)
            
            # Backup full array
            np.save(os.path.join(OUTPUT_DIR, base_name + "_all_rows.npy"), sig.astype(np.float32))

        except Exception as e:
            print(f"  Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if temp_path and os.path.exists(temp_path): 
                os.remove(temp_path)

if __name__ == "__main__":
    run_all()