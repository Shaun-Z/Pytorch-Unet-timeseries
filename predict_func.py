import pandas as pd
import argparse
import logging
from pathlib import Path
from utils.data_loading import SGCCDataset
import torch
from unet import UNet_1D, UNet_1D_N, UNet_1D_L, UNet_1D_NN, UNet_1D_LL
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_name = 'SGCC'
attack_id = 4
val_percent = 0.1
model_name = 'UNet_1D'
model = 'best_checkpoint.pth'
dir_checkpoint = Path(f'./checkpoints_pseudo_{data_name}/')

net = UNet_1D(n_channels=1, n_classes=2, bilinear=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
state_dict = torch.load(f'./checkpoints_pseudo_{data_name}/{model}', map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)

def predict_mask(data_val):
    if len(data_val.shape) == 2:
        data_val = data_val.unsqueeze(1)

    data_val = data_val.to(device)
    net.eval()
    with torch.no_grad():
        output = net(data_val)
        if net.n_classes > 1:
            mask_pred = output.argmax(dim=1)
        else:
            mask_pred = torch.sigmoid(output) > 0.5
    return mask_pred.cpu().numpy()

