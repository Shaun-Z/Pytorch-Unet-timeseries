from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, roc_auc_score
from utils.data_loading import SGCCDataset
import torch
from torch.utils.data import random_split
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

import numpy as np

# Load dataset
id = 'SGCC'
# dir_data = Path(f'./data/{id}_data/data_prepared/combined_dfx.csv')
# dir_mask = Path(f'./data/{id}_data/data_prepared/combined_dfy_pseudo.csv')
dir_data = Path(f'./data/{id}_data/data_prepared/test_data.csv')
dir_mask = Path(f'./data/{id}_data/data_prepared/test_label.csv')
dataset = SGCCDataset(dir_data, dir_mask, normalize=True)

# Split dataset into training and testing sets
val_percent: float = 0.1
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))


X_train = np.array([train_set[i]['data'] for i in range(len(train_set))]).squeeze()
y_train = np.array([train_set[i]['mask'] for i in range(len(train_set))])
y_train = (y_train.sum(axis=1) > 0).astype(int)
X_test = np.array([val_set[i]['data'] for i in range(len(val_set))]).squeeze()
y_test = np.array([val_set[i]['mask'] for i in range(len(val_set))])
y_test = (y_test.sum(axis=1) > 0).astype(int)

# X_train = train_set.dataset.data_tensor.squeeze()
# y_train = train_set.dataset.mask_tensor.squeeze()
# y_train = (y_train[..., :-12].sum(dim=1) > 0).long()

# X_test = val_set.dataset.data_tensor.squeeze()
# y_test = val_set.dataset.mask_tensor.squeeze()
# y_test = (y_test[..., :-12].sum(dim=1) > 0).long()

# print(X_train.shape, y_train.shape, y_train.unique(), X_test.shape, y_test.shape, y_test.unique())
print(X_train.shape, y_train.shape, np.unique(y_train), X_test.shape, y_test.shape, np.unique(y_test))

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt

# Plot X_train with colors based on y_train
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(X_train[y_train == 0].T, 'b', alpha=0.1)
plt.subplot(2, 1, 2)
plt.plot(X_train[y_train == 1].T, 'r', alpha=0.1)
plt.show()

# Create and train Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)
# y_pred_proba = dt_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print(classification_report(y_test, y_pred, digits=4))

# Calculate additional metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Recall: {recall}')
print(f'AUC: {auc}')