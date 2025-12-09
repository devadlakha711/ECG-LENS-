import ast
import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================
# CONFIG
# =========================
PTBXL_DIR = "ptbxl/ptb-xl1"   # change if your path is different
FS = 500              # Hz
DURATION = 10         # seconds
TARGET_LEN = FS * DURATION
N_LEADS = 12
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# STEP 1: LOAD METADATA
# =========================

db_path = os.path.join(PTBXL_DIR, "ptbxl_database.csv")
scp_path = os.path.join(PTBXL_DIR, "scp_statements.csv")

df = pd.read_csv(db_path)
scp_df = pd.read_csv(scp_path)

# stop if required columns not found
assert "scp_codes" in df.columns
assert "filename_hr" in df.columns

# Build full path to WFDB record (without extension)
df["path"] = df["filename_hr"].apply(lambda x: os.path.join(PTBXL_DIR, x))


# =========================
# STEP 2: DEFINE LABELS (MI, AF, AFL, OTHER_ARR)
# =========================

# The first column of scp_statements.csv holds the SCP code (NORM, IMI, AFIB, AFLT, ...)
code_col = scp_df.columns[0]   # should be something like 'Unnamed: 0'
print("Code column name:", code_col)

# (A) MI: all codes whose diagnostic_class == 'MI'
MI_CODES = set(
    scp_df.loc[scp_df["diagnostic_class"] == "MI", code_col].values.tolist()
)

# (B) AF: AFIB (no plain 'AF' in your file)
AF_CODES = {"AFIB"}

# (C) AFL: atrial flutter is coded as AFLT in your file
AFL_CODES = {"AFLT"}

# (D) OTHER_ARR: all other rhythm codes except AFIB/AFLT
# rhythm == 1 marks rhythm-related statements (SR, AFIB, AFLT, STACH, SBRAD, etc.)
rhythm_codes = set(
    scp_df.loc[scp_df["rhythm"] == 1, code_col].values.tolist()
)

# remove AF and AFL from OTHER_ARR group
other_arrhythmia_codes = rhythm_codes - AF_CODES - AFL_CODES

print("MI_CODES:", MI_CODES)
print("AF_CODES:", AF_CODES)
print("AFL_CODES:", AFL_CODES)
print("Some OTHER_ARR codes:", list(other_arrhythmia_codes)[:10])

LABELS = ["MI", "AF", "AFL", "OTHER_ARR"]
NUM_LABELS = len(LABELS)


def make_multilabel_target(scp_codes_str):
    """
    scp_codes_str: stringified dict, e.g. "{'NORM': 100.0, 'IMI': 80.0}"
    returns: np.array shape [4] with 0/1 for [MI, AF, AFL, OTHER_ARR]
    """
    d = ast.literal_eval(scp_codes_str)  # dict: code -> weight
    codes = set(d.keys())

    y = np.zeros(NUM_LABELS, dtype=np.float32)

    # MI: any code in MI_CODES
    if any(c in codes for c in MI_CODES):
        y[0] = 1.0

    # AF: AFIB
    if any(c in codes for c in AF_CODES):
        y[1] = 1.0

    # AFL: AFLT
    if any(c in codes for c in AFL_CODES):
        y[2] = 1.0

    # OTHER_ARR: any rhythm code except AFIB/AFLT
    if any(c in codes for c in other_arrhythmia_codes):
        y[3] = 1.0

    return y


df["y_vec"] = df["scp_codes"].apply(make_multilabel_target)
df["any_label"] = df["y_vec"].apply(lambda v: int(np.any(v)))



# =========================
# STEP 3: TRAIN/VAL/TEST SPLIT
# =========================

# Stratify on "any_label" so that abnormal / normal balanced across splits
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["any_label"],
    random_state=42
)
train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["any_label"],
    random_state=42
)

print("Train size:", len(train_df))
print("Val size  :", len(val_df))
print("Test size :", len(test_df))

# =========================
# STEP 4: PREPROCESSING – FILTERS
# =========================

def bandpass_filter(x, fs=FS, low=0.5, high=40.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def notch_filter(x, fs=FS, freq=50.0, q=30.0):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, x)

def preprocess_ecg(signal):
    """
    signal: np.ndarray [T, 12] from wfdb (time x leads)
    returns: np.ndarray [12, TARGET_LEN]
    """
    sig = signal.T.astype(np.float32)   # [12, T]
    T = sig.shape[1]

    # Ensure 10-second length (crop or pad)
    if T > TARGET_LEN:
        sig = sig[:, :TARGET_LEN]
    elif T < TARGET_LEN:
        pad_width = TARGET_LEN - T
        sig = np.pad(sig, ((0, 0), (0, pad_width)), mode="edge")

    # Filter & normalize per-lead
    for i in range(N_LEADS):
        lead = sig[i]
        # Bandpass
        lead = bandpass_filter(lead)
        # Notch (optional)
        lead = notch_filter(lead)
        # Z-score
        lead = (lead - lead.mean()) / (lead.std() + 1e-7)
        sig[i] = lead

    return sig  # [12, TARGET_LEN]


# =========================
# STEP 5: DATASET & DATALOADER
# =========================

class PTBXLMultilabelDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        y_vec = np.array(row["y_vec"], dtype=np.float32)  # [4]

        # read ECG with wfdb
        signal, fields = wfdb.rdsamp(path)  # signal: [T, 12]
        x = preprocess_ecg(signal)          # [12, TARGET_LEN]

        x = torch.from_numpy(x).float()     # [12, T]
        y = torch.from_numpy(y_vec).float() # [4]

        return x, y

train_dataset = PTBXLMultilabelDataset(train_df)
val_dataset   = PTBXLMultilabelDataset(val_df)
test_dataset  = PTBXLMultilabelDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS)


# =========================
# STEP 6: 1D CNN MODEL (MULTI-LABEL)
# =========================

class AFNet1D_Multi(nn.Module):
    def __init__(self, in_channels=12, num_labels=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm1d(128)

        self.pool  = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, num_labels)

    def forward(self, x):
        # x: [B, 12, T]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)    # [B, 128, 1]
        x = x.squeeze(-1)          # [B, 128]
        x = self.fc(x)             # [B, num_labels]
        return x


model = AFNet1D_Multi(in_channels=12, num_labels=NUM_LABELS).to(DEVICE)

# Multi-label BCE with logits
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# =========================
# STEP 7: EVALUATION FUNCTION
# =========================

def eval_model(model, loader):
    model.eval()
    all_targets = []
    all_probs   = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)            # [B, 4]
            probs  = torch.sigmoid(logits)

            all_targets.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)  # [N, 4]
    y_prob = np.concatenate(all_probs, axis=0)    # [N, 4]
    y_pred = (y_prob > 0.5).astype(int)

    # Compute per-label AUC and F1
    aucs = []
    f1s  = []
    for i, label_name in enumerate(LABELS):
        y_t = y_true[:, i]
        y_p = y_prob[:, i]
        y_hat = y_pred[:, i]

        # If no positives at all, skip AUC to avoid error
        if np.sum(y_t) == 0 or np.sum(1 - y_t) == 0:
            auc = np.nan
        else:
            auc = roc_auc_score(y_t, y_p)
        if np.sum(y_hat) == 0 and np.sum(y_t) == 0:
            f1 = 1.0
        else:
            f1 = f1_score(y_t, y_hat)

        aucs.append(auc)
        f1s.append(f1)

    return aucs, f1s


# =========================
# STEP 8: TRAINING LOOP
# =========================

best_val_macro_auc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
        x = x.to(DEVICE)       # [B, 12, T]
        y = y.to(DEVICE)       # [B, 4]

        optimizer.zero_grad()
        logits = model(x)      # [B, 4]
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    val_aucs, val_f1s = eval_model(model, val_loader)
    # macro AUC ignoring NaNs
    valid_aucs = [a for a in val_aucs if not np.isnan(a)]
    macro_auc = np.mean(valid_aucs) if len(valid_aucs) > 0 else np.nan
    macro_f1  = np.mean(val_f1s)

    print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}")
    for i, lab in enumerate(LABELS):
        print(f"  Val {lab}: AUC={val_aucs[i]:.4f}  F1={val_f1s[i]:.4f}")
    print(f"  Val macro AUC={macro_auc:.4f}, macro F1={macro_f1:.4f}")

    if not np.isnan(macro_auc) and macro_auc > best_val_macro_auc:
        best_val_macro_auc = macro_auc
        torch.save(model.state_dict(), "multilabel_af_mi_model.pt")
        print("  🔥 New best model saved.")


# =========================
# STEP 9: FINAL TEST EVAL
# =========================

model.load_state_dict(torch.load("multilabel_af_mi_model.pt", map_location=DEVICE))
test_aucs, test_f1s = eval_model(model, test_loader)

print("\n=== TEST RESULTS ===")
for i, lab in enumerate(LABELS):
    print(f"Test {lab}: AUC={test_aucs[i]:.4f}  F1={test_f1s[i]:.4f}")

valid_test_aucs = [a for a in test_aucs if not np.isnan(a)]
print(f"Test macro AUC={np.mean(valid_test_aucs):.4f}, macro F1={np.mean(test_f1s):.4f}")


# =========================
# STEP 10: PREDICTION HELPER FUNCTION
# =========================

def predict_labels_from_signal(ecg_12xT: np.ndarray) -> dict:
    """
    ecg_12xT: np.ndarray [12, T_raw] in mV (approx)
    returns: dict with probabilities for each label
    """
    # If input is [T, 12], transpose:
    if ecg_12xT.shape[0] != N_LEADS and ecg_12xT.shape[1] == N_LEADS:
        ecg_12xT = ecg_12xT.T

    # Convert to [T, 12] for preprocess
    signal = ecg_12xT.T  # [T, 12]
    x_pre = preprocess_ecg(signal)  # [12, TARGET_LEN]

    x_tensor = torch.from_numpy(x_pre).float().unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)  # [1, 4]
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return {label: float(p) for label, p in zip(LABELS, probs)}


# Example usage (once model is trained):
# ecg = ... # your 12xT signal array from digitizer
# preds = predict_labels_from_signal(ecg)
# print(preds)  # {"MI": 0.23, "AF": 0.91, "AFL": 0.02, "OTHER_ARR": 0.35}
