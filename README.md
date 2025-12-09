# ECG Image to Diagnosis Pipeline

This project builds an end-to-end pipeline that converts **ECG images → digitized signals → model predictions**.

The system performs the following steps:

1. **Digitize ECG images** into numerical signals
2. **Preprocess the signals**
3. **Convert signals to model input format**
4. **Run a trained deep learning model** to predict ECG abnormalities

The project uses the **PTB-XL dataset** for training and evaluation.

---

# Project Structure

```
project/
│
├── ecg_digitiser/              # ECG digitization library
├── ptbxl/                      # PTB-XL metadata
│   ├── ptbxl_database.csv
│   └── scp_statements.csv
│
├── classifier_wrapper.py
├── convert_to_model.py
├── digitizer_runner.py
├── image_to_prediction.py
├── inspect_hr.py
├── plot_digitised.py
├── predict_from_images.py
├── preprocess.py
├── reorganize.py
├── train_ptbxl_multilabel.py
├── verify_npy.py
│
├── requirements.txt
└── README.md
```

---

# Dataset Setup (PTB-XL)

Download the PTB-XL dataset from PhysioNet:

https://physionet.org/content/ptb-xl/1.0.3/

After downloading, place it inside the project directory:

```
ptbxl/
│
├── records100/
├── records500/
├── ptbxl_database.csv
└── scp_statements.csv
```

⚠️ The dataset is ~8GB and therefore **not included in this repository**.

---

# Environment Setup

It is recommended to run this project inside a **Python virtual environment**.

## 1. Create Virtual Environment

```bash
python -m venv venv
```

## 2. Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

# Install Required Python Libraries

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

---

# Generate `requirements.txt`

If you want to regenerate the dependency file after installing packages:

```bash
pip freeze > requirements.txt
```

This will save all installed Python libraries required to reproduce the environment.

---

# Verify Installation

Check installed packages:

```bash
pip list
```

---

# Important Notes

* The `venv/` folder should **NOT be uploaded to GitHub**.
* Add the following entry to `.gitignore`:

```
venv/
__pycache__/
*.pyc
```

---

# Clone ECG Digitizer Dependency

This project uses the **Open-ECG-Digitizer** library.

Clone it inside the project directory:

```bash
git clone https://github.com/Ahus-AIM/Open-ECG-Digitizer.git ecg_digitiser
```

---

# Apply Required Modifications

After cloning the repository, several files must be modified to integrate with this project.

The modified files and instructions are provided below.

(You will add the file list here.)

Example structure:

```
ecg_digitiser/
   file1.py
   file2.py
   module/
      file3.py
```

Replace the corresponding files with the modified versions in this repository.

---

# Next Steps

After setting up the environment and cloning the digitizer repository, you can proceed to run the ECG digitization and prediction pipeline.


# Clone ECG Digitizer

This project depends on the **Open-ECG-Digitizer** library.

Clone it inside the project directory:

```
git clone https://github.com/Ahus-AIM/Open-ECG-Digitizer.git ecg_digitiser
```

This will create the folder:

```
ecg_digitiser/
```

---

# Apply Required Code Modifications

After cloning the digitizer repository, replace the following files with the modified versions provided in this project.

Modified files:

## Modify `ecg_digitiser/run_digitizer.py`

```text
ecg_digitiser/run_digitizer.py
```

```python
import os
import sys
import glob
import yaml
import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image

_this_dir = os.path.dirname(__file__)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

try:
    from src.model.inference_wrapper import InferenceWrapper
except Exception as e:
    raise RuntimeError("Cannot import InferenceWrapper.") from e

def _find_config_file():
    cfg_dir = os.path.join(os.path.dirname(__file__), "src", "config")
    cand = []
    for ext in ("*.yml", "*.yaml"):
        cand.extend(sorted(glob.glob(os.path.join(cfg_dir, ext))))
    if not cand:
        raise RuntimeError(f"No yaml config files found")
    for path in cand:
        if any(x in os.path.basename(path).lower() for x in ("inference", "infer", "deploy")):
            return path
    return cand[0]

def _load_cfgnode_from_yaml(path):
    from yacs.config import CfgNode as CN
    return CN(yaml.safe_load(open(path, "r", encoding="utf-8")))

def load_png_file(path):
    img = read_image(path).float() / 255.0
    if img.shape[0] > 3: img = img[:3, :, :]
    return img.unsqueeze(0)

def digitize_image_from_path(seg_weights_path, layout_weights_path, image_path,
                             resample_size=2000, target_num_samples=5000, device=None,
                             layout_substring=None):
    if device is None: device = torch.device("cpu")
    elif isinstance(device, str): device = torch.device(device)

    cfg_path = _find_config_file()
    cfg = _load_cfgnode_from_yaml(cfg_path)
    
    from yacs.config import CfgNode as CN
    if not hasattr(cfg, "MODEL"): cfg.MODEL = CN()
    if not hasattr(cfg.MODEL.KWARGS, "config"): cfg.MODEL.KWARGS.config = CN()
    nested_cfg = cfg.MODEL.KWARGS.config

    for sub in ("SEGMENTATION_MODEL", "LAYOUT_IDENTIFIER", "SIGNAL_EXTRACTOR", 
                "CROPPER", "PIXEL_SIZE_FINDER", "DEWARPER", "PERSPECTIVE_DETECTOR"):
        if not hasattr(nested_cfg, sub): setattr(nested_cfg, sub, CN())

    nested_cfg.SEGMENTATION_MODEL.weight_path = seg_weights_path
    nested_cfg.LAYOUT_IDENTIFIER.unet_weight_path = layout_weights_path

    base = os.path.dirname(__file__)
    def _resolve_path(p):
        if not isinstance(p, str): return p
        if os.path.exists(p): return os.path.abspath(p)
        cand = os.path.join(base, "src", "config", os.path.basename(p))
        if os.path.exists(cand): return cand
        return p

    try:
        if hasattr(nested_cfg, "LAYOUT_IDENTIFIER"):
            li = nested_cfg.LAYOUT_IDENTIFIER
            for attr in ("config_path", "unet_config_path", "unet_weight_path"):
                if hasattr(li, attr): setattr(li, attr, _resolve_path(getattr(li, attr)))
        if hasattr(nested_cfg, "SEGMENTATION_MODEL"):
            nested_cfg.SEGMENTATION_MODEL.weight_path = _resolve_path(nested_cfg.SEGMENTATION_MODEL.weight_path)
    except:
        pass

    wrapper = InferenceWrapper(nested_cfg, device=device, resample_size=resample_size, enable_timing=False)
    wrapper = wrapper.to(device).eval()

    img = load_png_file(image_path)
    
    with torch.no_grad():
        out = wrapper(img.to(device), layout_should_include_substring=layout_substring)

    signal_section = out.get("signal", {})
    sig = None
    for k in ("canonical_lines", "lines", "raw_lines"):
        if signal_section.get(k) is not None:
            sig = signal_section.get(k)
            break

    if sig is None: raise RuntimeError("InferenceWrapper returned no signal lines.")
    if torch.is_tensor(sig): sig = sig.cpu().numpy()

    if sig.shape[0] > sig.shape[1]: 
        sig = sig.T

    sig = sig * 0.001
    for i in range(sig.shape[0]):
        series = pd.Series(sig[i])
        series = series.interpolate(method='linear', limit_direction='both')
        series = series.fillna(0)
        sig[i] = series.to_numpy()

    return sig.astype(np.float32)
```


Steps:

1. Navigate to the `ecg_digitiser` directory
2. Replace the corresponding files with the modified versions
3. Save the changes

Example:

```
cp modified_file.py ecg_digitiser/path/to/file.py
```

These modifications adapt the digitizer to work with the **ECG image processing pipeline used in this project**.

---

# Pipeline Overview

The workflow is:

```
ECG Image
     ↓
Digitization (Open-ECG-Digitizer)
     ↓
Signal preprocessing
     ↓
Conversion to model input
     ↓
Deep learning model prediction
```

---

# Running the Pipeline

Digitize ECG images:

```
python digitizer_runner.py
```

Convert digitized signals for model input:

```
python convert_to_model.py
```

Run predictions:

```
python predict_from_images.py
```

---

# Training the Model

To train the PTB-XL multilabel classifier:

```
python train_ptbxl_multilabel.py
```

---

# Acknowledgements

* PTB-XL Dataset (PhysioNet)
* Open-ECG-Digitizer
* WFDB library



