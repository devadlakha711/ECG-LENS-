# predict_from_images.py
import os
from ecg_digitiser.run_digitizer import load_segmentation_unet, load_layout_unet, digitize_image_from_path
from classifier_wrapper import load_classifier, predict_labels_from_signal
from preprocess import preprocess_ecg
from preprocess import TARGET_LEN
import numpy as np

INPUT_DIR = "ecg_images"
RESAMPLE_SIZE = 2000
TARGET_NUM_SAMPLES = TARGET_LEN

def main():
    classifier = load_classifier()
    seg_unet = load_segmentation_unet()
    layout_unet = load_layout_unet()

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))]
    print("Found", len(files), "images.")
    for f in files:
        path = os.path.join(INPUT_DIR, f)
        print("\nProcessing:", path)
        try:
            sig = digitize_image_from_path(seg_unet, layout_unet, path, resample_size=RESAMPLE_SIZE, target_num_samples=TARGET_NUM_SAMPLES)
        except Exception as e:
            print(" Digitizer error:", e)
            continue

        if sig.shape[0] != 12 and sig.shape[1] == 12:
            sig = sig.T

        # if len not equal, resample/pad to target
        if sig.shape[1] != TARGET_NUM_SAMPLES:
            from preprocess import resample_to_target, pad_or_crop
            sig = resample_to_target(sig, TARGET_NUM_SAMPLES)
            sig = pad_or_crop(sig, TARGET_NUM_SAMPLES)

        print(" Signal shape:", sig.shape, "min/max:", np.nanmin(sig), np.nanmax(sig))
        preds = predict_labels_from_signal(sig, classifier, preprocess_ecg)
        print(" Predictions:", preds)

if __name__ == "__main__":
    main()
