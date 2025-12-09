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

```
FILE_1
FILE_2
FILE_3
FILE_4
FILE_5
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



