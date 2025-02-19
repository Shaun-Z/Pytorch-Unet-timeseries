import pandas as pd
import logging
from pathlib import Path
from utils.data_loading import SGCCDataset
import torch
from unet import UNet_1D

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

zx3_normalized = pd.read_csv(f'./zx3_normalized.csv') 
normal3_normalized = pd.read_csv(f'./Normal3_normalized.csv')
# Rename the columns of df2 to match df1
normal3_normalized.columns = zx3_normalized.columns

zy3 = pd.read_csv(f'./zy3.csv') 
normal3_normalized_label = pd.read_csv(f'./Normal3_normalized_label.csv')
normal3_normalized_label.columns = zy3.columns

combined_dfx = pd.concat([zx3_normalized, normal3_normalized], ignore_index=True)#
combined_dfy = pd.concat([zy3, normal3_normalized_label], ignore_index=True)#


combined_dfx.to_csv('./combined_dfx.csv', index=False)
combined_dfy.to_csv('./combined_dfy.csv', index=False)


normal3_normalized_sudolabel = pd.read_csv(f'./Normal3_normalized_sudolabel.csv')
normal3_normalized_sudolabel.columns = zy3.columns
combined_dfy_sudo = pd.concat([zy3, normal3_normalized_sudolabel], ignore_index=True)#
combined_dfy_sudo.to_csv('./combined_dfy_sudo.csv', index=False)

normal3_normalized_sudolabel1 = pd.read_csv(f'./Normal3_normalized_sudolabel1.csv')
normal3_normalized_sudolabel1.columns = zy3.columns
combined_dfy_sudo1 = pd.concat([zy3, normal3_normalized_sudolabel1], ignore_index=True)#
combined_dfy_sudo1.to_csv('./combined_dfy_sudo1.csv', index=False)

normal3_normalized_sudolabel2 = pd.read_csv(f'./Normal3_normalized_sudolabel2.csv')
normal3_normalized_sudolabel2.columns = zy3.columns
combined_dfy_sudo2 = pd.concat([zy3, normal3_normalized_sudolabel2], ignore_index=True)#
combined_dfy_sudo2.to_csv('./combined_dfy_sudo2.csv', index=False)

# exit()
#!python Pytorch-Unet-timeseries/train_t.py


##### testing 用的predict_t.py
id = '2'
##### 改变的地方是：
dir_data = Path('./combined_dfx.csv')
# dir_mask = Path('./combined_dfy.csv')
dir_mask = Path(f'./combined_dfy_sudo{id}.csv')
dataset = SGCCDataset(dir_data, dir_mask)

data = dataset.data_tensor
mask = dataset.mask_tensor


net = UNet_1D(n_channels=1, n_classes=2, bilinear=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net.to(device=device)
# state_dict = torch.load('./checkpoints/checkpoint_epoch21.pth', map_location=device)
# state_dict = torch.load('./checkpoints_Attack3_pure/checkpoint_epoch20.pth', map_location=device)##old attack 3
# state_dict = torch.load('./checkpoints_Attack3_pure/checkpoint_epoch20_combined.pth', map_location=device)##old attack 3+new 3000 normal data
# state_dict = torch.load('./checkpoints_Attack3_pure/checkpoint_epoch20_combine_sudo.pth', map_location=device)##old attack 3+new 3000 normal data+sudo label
state_dict = torch.load(f'./checkpoints_sudo_{id}/checkpoint_epoch20.pth', map_location=device)##old attack 3+new 3000 normal data+sudo label
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

# zy = pd.read_csv('./combined_dfy.csv')
zy = pd.read_csv(f'./combined_dfy_sudo{id}.csv')
# zy = pd.read_csv('.\Attack3_normalized_label.csv')


import matplotlib.pyplot as plt
import seaborn as sns
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle(f'Attack 3')   

# Plot the first heatmap
sns.heatmap(result_df, ax=ax1)
ax1.set_title('Heatmap of prediction')

# Plot the second heatmap
sns.heatmap(zy, ax=ax2)
ax2.set_title('Heatmap of target')

# Display the plot
plt.tight_layout()
# plt.savefig(f'result{id}.png')
plt.show()
