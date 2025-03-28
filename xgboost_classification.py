from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, roc_auc_score
from utils.data_loading import SGCCDataset
import torch
from torch.utils.data import random_split
from pathlib import Path
import xgboost as xgb

import numpy as np
from sklearn.metrics import log_loss

# Load dataset
id = 'SGCC'
dir_data = Path(f'./data/{id}_data/data_prepared_6/combined_dfx.csv')
dir_mask = Path(f'./data/{id}_data/data_prepared_6/combined_dfy.csv')
dataset = SGCCDataset(dir_data, dir_mask, normalize=True)

# Split dataset into training and testing sets
val_percent: float = 0.3
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

X_train = np.array([train_set[i]['data'] for i in range(len(train_set))]).squeeze()
y_train = np.array([train_set[i]['mask'] for i in range(len(train_set))])
y_train = (y_train[..., :-12].sum(axis=1) > 0).astype(int)
X_test = np.array([val_set[i]['data'] for i in range(len(val_set))]).squeeze()
y_test = np.array([val_set[i]['mask'] for i in range(len(val_set))])
y_test = (y_test[..., :-12].sum(axis=1) > 0).astype(int)

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

# Create and train XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Calculate cross entropy loss on training set
y_train_pred_proba = xgb_model.predict_proba(X_train)[:, 1]
cross_entropy_loss = log_loss(y_train, y_train_pred_proba)

print(f'Cross Entropy Loss on Training Set: {cross_entropy_loss}')

y_test_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
cross_entropy_loss = log_loss(y_test, y_test_pred_proba)

print(f'Cross Entropy Loss on Test Set: {cross_entropy_loss}')

# Make predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print(classification_report(y_test, y_pred, digits=4))

# Calculate additional metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Recall: {recall}')
print(f'AUC: {auc}')