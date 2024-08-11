# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:37:48 2024

@author: 37092
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from pathlib import Path
from utils.data_loading import BasicDataset
from unet import UNet_1D
from utils.utils import plot_img_and_mask

from utils.data_loading import SGCCDataset
import pandas as pd
from torch.utils.data import DataLoader, random_split

def predict_data(net,
                data,
                device,
                out_threshold=0.5):
    net.eval()
    data = data.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(data).cpu()
        # print(output.size())
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
        # print(mask.size())
    return mask[0].long().squeeze().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)    
    
id_model = '3'
id_data_test = '32000'

# dir_data = Path('./data/attack.csv')
# dir_mask = Path('./data/label.csv')
# dir_data = Path('./data_add_noise/zx.csv')
dir_data = Path(f'./zx{id_data_test}_normalized.csv')
# dir_data = Path('./data_add_noise/usable_theft_2016_linear.csv')######693 out of 1989 wrong
dir_mask = Path(f'./zy{id_data_test}.csv')
dataset = SGCCDataset(dir_data, dir_mask)
data = dataset.data_tensor
mask = dataset.mask_tensor


net = UNet_1D(n_channels=1, n_classes=2, bilinear=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net.to(device=device)
state_dict = torch.load(f'./checkpoints{id_model}/checkpoint_epoch40.pth', map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)

result = []
pred_ts = []
for i in range(len(dataset)):
    logging.info(f'Predicting No. {i} ...')
    item = data[i,:,:].unsqueeze(0)  
    
    mask = predict_data(net=net,
                        data=item,
                        out_threshold=0.5,
                        device=device)

    result.append(mask)

result_df = pd.DataFrame(result)
# result_df.to_csv('result.csv', index=False)

zy = pd.read_csv(f'./zy{id_data_test}.csv')





import matplotlib.pyplot as plt
import seaborn as sns
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle(f'Attack {id_model}')   

# Plot the first heatmap
sns.heatmap(result_df, ax=ax1)
ax1.set_title('Heatmap of prediction')

# Plot the second heatmap
sns.heatmap(zy, ax=ax2)
ax2.set_title('Heatmap of target')

# Display the plot
plt.tight_layout()
plt.savefig(f'result{id_model}_mix.png')
plt.show()



result_df_sum = result_df.sum(axis=1)




    