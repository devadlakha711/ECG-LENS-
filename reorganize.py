import numpy as np
import scipy.signal
import os
import glob
import matplotlib.pyplot as plt
import shutil

# === CONFIGURATION ===
INPUT_DIR = "digitized_signals"       
OUTPUT_DIR = "final_12x1250_leads"    
TARGET_SAMPLES = 1250                 

# Standard Lead Names matching indices 0-11
LEAD_NAMES = {
    0: "I",   1: "II",  2: "III",
    3: "aVR", 4: "aVL", 5: "aVF",
    6: "V1",  7: "V2",  8: "V3",
    9: "V4",  10: "V5", 11: "V6"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_active_rows(data):
    """
    indices of rows that have ANY signal data.
    """
    active_indices = []
    print("  > Checking row activity levels:")
    
    for i in range(data.shape[0]):
        row_data = data[i]
        amplitude = np.max(row_data) - np.min(row_data)
        
        if amplitude > 1e-8:
            active_indices.append(i)
            print(f"    Row {i}: ACTIVE (Amp: {amplitude:.6f})")
            
    return active_indices

def save_all_leads_plots(matrix, base_name):
    """
    Save 12 individual PNG plots for each lead.
    """
    # 1. idhar avoid clutter
    plot_folder = os.path.join(OUTPUT_DIR, f"{base_name}_plots")
    if os.path.exists(plot_folder):
        shutil.rmtree(plot_folder)
    os.makedirs(plot_folder, exist_ok=True)
    
    print(f"  > Saving 12 lead images to: {plot_folder}")

    # 2. Iterate and Plot
    for i in range(12):
        lead_name = LEAD_NAMES[i]
        signal = matrix[i]
        
        plt.figure(figsize=(10, 3))
        plt.plot(signal, color='black', linewidth=1.2)
        plt.title(f"{base_name} - Lead {lead_name}")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        filename = f"{lead_name}.png"
        save_path = os.path.join(plot_folder, filename)
        plt.savefig(save_path)
        plt.close()

def process_file(file_path):
    base_name = os.path.basename(file_path).replace("_all_rows.npy", "")
    print(f"Processing: {base_name}...")
    
    try:
        data = np.load(file_path)
        
        # 1.ACTIVE ROWS
        active_rows = find_active_rows(data)
        
        if len(active_rows) < 3:
            print(f"SKIP: {base_name} - Found only {len(active_rows)} active rows. Needed 3.")
            return
            
        target_row_indices = active_rows[:3]
        print(f"  > Using rows: {target_row_indices}")
        
        waveform_rows = data[target_row_indices]
        
        # 2. CREATE EMPTY MATRIX
        final_matrix = np.zeros((12, TARGET_SAMPLES), dtype=np.float32)
        
        # 3. SPLIT AND ASSIGN
        lead_map = [
            [0, 3, 6, 9],   # 1st active row -> I, aVR, V1, V4
            [1, 4, 7, 10],  # 2nd active row -> II, aVL, V2, V5
            [2, 5, 8, 11]   # 3rd active row -> III, aVF, V3, V6
        ]
        
        for i in range(3):
            full_row = waveform_rows[i]
            chunk_size = len(full_row) // 4
            
            for k in range(4):
                start = k * chunk_size
                end = start + chunk_size
                segment = full_row[start:end]
                
                # Resample to exactly 1250
                if len(segment) != TARGET_SAMPLES:
                    segment = scipy.signal.resample(segment, TARGET_SAMPLES)
                
                lead_index = lead_map[i][k]
                final_matrix[lead_index] = segment

        # 4. SAVE NPY
        npy_path = os.path.join(OUTPUT_DIR, f"{base_name}.npy")
        np.save(npy_path, final_matrix)
        print(f"  ✅ Saved NPY: {npy_path}")
        
        # 5. SAVE 12 IMAGES
        save_all_leads_plots(final_matrix, base_name)

    except Exception as e:
        print(f"erROR processing {base_name}: {e}")
        import traceback
        traceback.print_exc()

def run_batch():
    search_pattern = os.path.join(INPUT_DIR, "*_all_rows.npy")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files found in {INPUT_DIR} matching '*_all_rows.npy'")
        return

    print(f"Found {len(files)} files.")
    print("-" * 50)
    
    for f in files:
        process_file(f)
        print("-" * 50)

if __name__ == "__main__":
    run_batch()