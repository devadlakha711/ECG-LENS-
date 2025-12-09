import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os
import glob

# === CONFIGURATION ===
INPUT_DIR = "digitized_signals"       # Where the _all_rows.npy files are
OUTPUT_DIR = "final_model_input"      # New folder for output
TARGET_COLS = 250                     # Target width for the model

os.makedirs(OUTPUT_DIR, exist_ok=True)

def resize_signal(signal, target_length):
    """Resamples a 1D signal to a specific length."""
    return scipy.signal.resample(signal, target_length)

def create_preview_image(stitched_rows, base_name, save_path):
    """
    Creates a PNG showing the 3 main rows (Waveforms 1, 2, 3).
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    
    # Titles for the 3 rows based on your description
    titles = [
        "Waveform 1 (I, aVR, V1, V4)",
        "Waveform 2 (II, aVL, V2, V5)",
        "Waveform 3 (III, aVF, V3, V6)"
    ]

    for i in range(3):
        ax = axes[i]
        ax.plot(stitched_rows[i], color='black', linewidth=1.2)
        ax.set_title(titles[i], fontweight='bold', loc='left', fontsize=10)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel("Amplitude", fontsize=8)

    plt.suptitle(f"Preview: {base_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    plt.savefig(save_path, dpi=150)
    plt.close()

def process_file(file_path):
    base_name = os.path.basename(file_path).replace("_all_rows.npy", "")
    print(f"Processing: {base_name}...")
    
    try:
        data = np.load(file_path)
        
        # Check validation
        if data.shape[0] < 3:
            print(f"  ⚠️ SKIP: {base_name} has fewer than 3 rows ({data.shape[0]}).")
            return

        # 1. EXTRACT 3 ROWS (Discard Rhythm strip if present)
        stitched_rows = data[0:3] 
        current_length = stitched_rows.shape[1]
        
        # --- GENERATE PREVIEW IMAGE ---
        png_path = os.path.join(OUTPUT_DIR, f"{base_name}_preview.png")
        create_preview_image(stitched_rows, base_name, png_path)
        print(f"  📸 Saved Preview: {png_path}")

        # --- CONVERT TO 12x250 FOR MODEL ---
        chunk_size = current_length // 4
        final_matrix = np.zeros((12, TARGET_COLS), dtype=np.float32)

        lead_indices = [
            [0, 3, 6, 9],   # Row 0 -> I, aVR, V1, V4
            [1, 4, 7, 10],  # Row 1 -> II, aVL, V2, V5
            [2, 5, 8, 11]   # Row 2 -> III, aVF, V3, V6
        ]

        for row_idx in range(3):
            source_row = stitched_rows[row_idx]
            for i in range(4):
                start = i * chunk_size
                end = start + chunk_size
                if i == 3: segment = source_row[start:]
                else: segment = source_row[start:end]
                
                # Resize and store
                resized_segment = resize_signal(segment, TARGET_COLS)
                target_lead_idx = lead_indices[row_idx][i]
                final_matrix[target_lead_idx] = resized_segment

        # Save NPY
        npy_path = os.path.join(OUTPUT_DIR, f"{base_name}.npy")
        np.save(npy_path, final_matrix)
        print(f"  ✅ Saved Data:    {npy_path}")

    except Exception as e:
        print(f"  ❌ ERROR processing {base_name}: {e}")

def run_batch_conversion():
    search_pattern = os.path.join(INPUT_DIR, "*_all_rows.npy")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files found in {INPUT_DIR} ending with '_all_rows.npy'")
        return

    print(f"Found {len(files)} files to convert.")
    print("-" * 50)
    
    for f in files:
        process_file(f)
        print("-" * 50)
        
    print("Batch conversion complete.")

if __name__ == "__main__":
    run_batch_conversion()