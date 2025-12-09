import torch, numpy as np
from scipy.signal import resample
# import your digitizer functions / UNet wrapper, and your classifier + preprocess_ecg()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_LEN = 5000  # 10s @ 500 Hz

def run_digitizer_on_image(image_path, seg_unet, layout_unet, other_models, resample_size=2000):
    # This calls the digitize_image pipeline (adapted from the Kaggle script)
    img = load_png_file(image_path)  # from their code
    output_probs, aligned_signal, aligned_grid, lines = digitize_image(img, resample_size, target_num_samples=TARGET_LEN)
    # lines is probably a tensor; convert to numpy
    sig = lines.cpu().numpy()   # expect shape (12, N)
    return sig

def postprocess_and_predict(sig, classifier_model):
    # 1) check shape and transpose if needed
    if sig.shape[0] != 12 and sig.shape[1] == 12:
        sig = sig.T

    # 2) resample/crop/pad to TARGET_LEN
    if sig.shape[1] != TARGET_LEN:
        sig = resample_to_target(sig, TARGET_LEN)

    sig = pad_or_crop(sig, TARGET_LEN)

    # 3) units sanity check: ensure mV scale
    print("Signal min/max (mV):", sig.min(), sig.max())

    # 4) preprocess (expects [T,12])
    signal_Tx12 = sig.T  # if your preprocess_ecg expects [T, 12]
    x_pre = preprocess_ecg(signal_Tx12)  # returns [12, TARGET_LEN]

    # 5) model inference
    x_tensor = torch.from_numpy(x_pre).float().unsqueeze(0).to(DEVICE)
    classifier_model.eval()
    with torch.no_grad():
        logits = classifier_model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return {label: float(p) for label,p in zip(LABELS, probs)}

# Usage
# load segmentation models (seg_unet, layout_unet etc.) per their code
# load your classifier: classifier_model.load_state_dict(torch.load("multilabel_af_mi_model.pt"))

sig = run_digitizer_on_image("ecg_001.png", seg_unet, layout_unet, ...)
preds = postprocess_and_predict(sig, classifier_model)
print(preds)
