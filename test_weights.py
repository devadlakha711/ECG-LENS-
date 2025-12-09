import torch

try:
    state = torch.load("ecg_digitiser/weights/unet_weights_07072025.pt", map_location="cpu")
    print("Loaded successfully. Number of parameters:", len(state))
except Exception as e:
    print("Error loading weights:", e)
