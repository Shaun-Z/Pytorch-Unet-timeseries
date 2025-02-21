import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

class Wide_CNN(nn.Module):
    def __init__(self, weeks, days, channel, wide_len):
        super(Wide_CNN, self).__init__()

        # Deep component
        self.conv = nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=3, stride=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3)
        
        # Dynamically compute the output size of the fc_deep layer
        sample_input = torch.zeros(1, channel, weeks, days)  # Sample input to determine shape
        with torch.no_grad():
            output_features = self.pool(self.conv(sample_input)).view(-1).size(0)

        self.flatten = nn.Flatten()
        self.fc_deep = nn.Linear(output_features, 128)

        # Wide component
        self.fc_wide = nn.Linear(wide_len, 128)

        # Combined layers
        self.fc_combined1 = nn.Linear(128 + 128, 64)
        self.fc_combined2 = nn.Linear(64, 1)

    def forward(self, inputs_wide, inputs_deep):
        # Deep pathway
        x_deep = self.conv(inputs_deep)
        x_deep = self.pool(x_deep)
        x_deep = self.flatten(x_deep)
        x_deep = F.relu(self.fc_deep(x_deep))

        # Wide pathway
        x_wide = F.relu(self.fc_wide(inputs_wide))

        # Combine pathways
        x = torch.cat((x_wide, x_deep), dim=1)
        x = F.relu(self.fc_combined1(x))
        pred = torch.sigmoid(self.fc_combined2(x))

        return pred

# Model initialization
def create_model(weeks, days, channel, wide_len, lr=0.005, decay=1e-5, momentum=0.9):
    model = Wide_CNN(weeks, days, channel, wide_len)

    # SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=momentum, nesterov=True)

    # Binary Cross Entropy Loss
    criterion = nn.BCELoss()

    return model, optimizer, criterion

def expand_data(data):      
    d1 = np.zeros([data.shape[0]*3, data.shape[1], data.shape[2]])
    
    for i in range(data.shape[0]):
        if i >= (data.shape[0] - 2):
            d1[i*3:(i*3+data.shape[0]-i),:,:] = data[i:,:,:]         
        else:
            d1[i*3:(i*3)+3,:,:] = data[i:i+3,:,:]
            
    
    d2 = np.zeros([d1.shape[0], d1.shape[1]*3, d1.shape[2]])
                  
    for j in range(d1.shape[1]):
        if j >= (d1.shape[1] - 2):
            d2[:,j*3:(j*3+d1.shape[1]-j),:] = d1[:,j:,:]         
        else:
            d2[:,j*3:(j*3)+3,:] = d1[:,j:(j+3),:]
    return d2
       
def preprocess_kernel(data):
    data1 = np.zeros(data.shape)
    data2 = np.zeros(data.shape)
    
    for i in range(int(data.shape[0]/3)):
        k = data[(i*3):(i*3+3),:,:]
        data1[i*3,:,:] = 2*k[0,:,:] - k[1,:,:] - k[2,:,:] 
        data1[i*3+1,:,:] = 2*k[1,:,:] - k[0,:,:] - k[2,:,:]
        data1[i*3+2,:,:] = 2*k[2,:,:] - k[0,:,:] - k[1,:,:]
    
    for i in range(int(data.shape[1]/3)):
        k = data[:,(i*3):(i*3+3),:]
        data1[:,i*3,:] = 2*k[:,0,:] - k[:,1,:] - k[:,2,:] 
        data1[:,i*3+1,:] = 2*k[:,1,:] - k[:,0,:] - k[:,2,:]
        data1[:,i*3+2,:] = 2*k[:,2,:] - k[:,0,:] - k[:,1,:]
    
    return data1 + data2

def self_define_cnn_kernel_process(data):
    '''
    1. expand data from (x, y, z) to (x*3, y*3, z) (Because Conv2D convolution with stride (3,3) for our preprocess)
    
    2. 3*3 kernel process:
    
        [2*V_1 - V_2 - V3
         2*V_2 - V_1 - V3
         2*V_3 - V_1 - V2]

        +

        [2*Vt_1 - Vt_2 - Vt_3, 2*Vt_2 - Vt_1 - Vt_3, 2*Vt_3 - Vt_1 - Vt_2]
      
    '''
    #input
    data_final = np.zeros([data.shape[0], data.shape[1]*3, data.shape[2]*3, data.shape[3]])
    for i in range(data.shape[0]):
        d1 = data[i,:,:,:]
        d1_expand = expand_data(d1)
        d1_final = preprocess_kernel(d1_expand)
        data_final[i,:,:,:] = d1_final
    print(data_final.shape)
    return data_final

if __name__ == '__main__':
    from pathlib import Path
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import f1_score, recall_score
    id = 'SGCC'
    dir_data = Path(f'./data/{id}_data/data_prepared/combined_dfx.csv')
    dir_mask = Path(f'./data/{id}_data/data_prepared/combined_dfy_pseudo.csv')
    dir_checkpoint = Path(f'./checkpoints_pseudo_{id}/')

    batch_size: int = 200
    epochs: int = 120
    learning_rate: float = 1e-4 # 1e-5
    save_checkpoint: bool = True
    amp: bool = False
    weight_decay: float = 1e-8
    momentum: float = 0.999
    gradient_clipping: float = 1.0
    weeks = 43  #43, 29
    days = 7   #7, 24

    data = pd.read_csv(dir_data)
    label = pd.read_csv(dir_mask)

    # Sum up each row of label and mark as 1 if greater than 14, else 0
    label = (label.iloc[:,:-12].sum(axis=1) > 0).astype(int)

    # for valr in [0.7, 0.6, 0.5]:
    valr = 0.9
    print('Train split ratio:%.2f'%valr)

    X_train_wide, X_test_wide, Y_train, Y_test = train_test_split(data.values, label.values, test_size=1-valr, random_state = 2017)

    # Y_train = Y_train[:Y_train.shape[0] - (Y_train.shape[0] % 7)]
    # Y_test = Y_test[:Y_test.shape[0] - (Y_test.shape[0] % 7)]
    # Trim the second dimension of X_train_wide and X_test_wide to be a multiple of 7
    X_train_wide = X_train_wide[:, :X_train_wide.shape[1] - (X_train_wide.shape[1] % days)]
    X_test_wide = X_test_wide[:, :X_test_wide.shape[1] - (X_test_wide.shape[1] % days)]
  

    X_train_deep = X_train_wide.reshape(X_train_wide.shape[0],1,-1,days)#.transpose(0,2,3,1)
    X_test_deep = X_test_wide.reshape(X_test_wide.shape[0],1,-1,days)#.transpose(0,2,3,1)

    print(X_train_wide.shape, X_train_deep.shape)
    print(X_test_wide.shape, X_test_deep.shape)

    # X_train_pre = self_define_cnn_kernel_process(X_train_deep)
    # X_test_pre = self_define_cnn_kernel_process(X_test_deep)
    X_train_pre = X_train_deep
    X_test_pre = X_test_deep

    # Initialize model
    model, optimizer, criterion = create_model(weeks=weeks, days=days, channel=1, wide_len=X_train_wide.shape[1], lr=learning_rate, decay=weight_decay, momentum=momentum)
    # weeks=43, days=7
    # day = 24, weeks = 29
    # Print model summary
    print(model)

    # Convert data to PyTorch tensors
    X_train_wide_tensor = torch.tensor(X_train_wide, dtype=torch.float32)
    X_train_pre_tensor = torch.tensor(X_train_pre, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

    X_test_wide_tensor = torch.tensor(X_test_wide, dtype=torch.float32)
    X_test_pre_tensor = torch.tensor(X_test_pre, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_wide_tensor, X_train_pre_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_wide_tensor, X_test_pre_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            inputs_wide, inputs_deep, targets = batch
            optimizer.zero_grad()

            outputs = model(inputs_wide, inputs_deep)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}')

    # Evaluation loop
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs_wide, inputs_deep, targets = batch
            outputs = model(inputs_wide, inputs_deep)
            loss = criterion(outputs.squeeze(), targets)
            test_loss += loss.item()
    
    print(f'Test Loss: {test_loss / len(test_loader)}')

    # Calculate accuracy on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs_wide, inputs_deep, targets = batch
            outputs = model(inputs_wide, inputs_deep)
            predicted = (outputs.squeeze() > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    
    # Calculate AUC on test set
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for batch in test_loader:
            inputs_wide, inputs_deep, targets = batch
            outputs = model(inputs_wide, inputs_deep)
            all_targets.extend(targets.numpy())
            all_outputs.extend(outputs.squeeze().numpy())

    auc = roc_auc_score(all_targets, all_outputs)
    # Calculate F1 and recall on test set
    f1 = f1_score(all_targets, (np.array(all_outputs) > 0.5).astype(int))
    recall = recall_score(all_targets, (np.array(all_outputs) > 0.5).astype(int))

    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1}')
    print(f'Recall: {recall}')
    print(f'AUC: {auc}')

    # Save the model checkpoint
    if save_checkpoint:
        checkpoint_path = Path(f'./checkpoints_pseudo_{id}/model_checkpoint.pth')
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)