# classifier_wrapper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LABELS = ["MI","AF","AFL","OTHER_ARR"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/multilabel_af_mi_model.pt"  # put your trained file here

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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

def load_classifier(model_path=MODEL_PATH):
    model = AFNet1D_Multi(in_channels=12, num_labels=len(LABELS)).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    # If saved as {'model_state_dict': ... } or similar, try to find nested
    if isinstance(state, dict):
        for candidate in ("state_dict", "model_state_dict", "model"):
            if candidate in state and isinstance(state[candidate], dict):
                state = state[candidate]
                break
    model.load_state_dict(state)
    model.eval()
    return model

def predict_labels_from_signal(sig_12xT_np, classifier_model, preprocess_fn):
    """
    sig_12xT_np: np.ndarray [12, T_raw] in mV
    preprocess_fn: function expecting [T,12] and returning [12, 5000]
    """
    if sig_12xT_np.shape[0] != 12 and sig_12xT_np.shape[1] == 12:
        sig_12xT_np = sig_12xT_np.T
    signal_Tx12 = sig_12xT_np.T  # [T,12]
    x_pre = preprocess_fn(signal_Tx12)  # [12, 5000]
    x_tensor = torch.from_numpy(x_pre).float().unsqueeze(0).to(DEVICE)  # [1,12,5000]
    with torch.no_grad():
        logits = classifier_model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return {label: float(p) for label,p in zip(LABELS, probs)}
