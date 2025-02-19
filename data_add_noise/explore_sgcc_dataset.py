# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:28:30 2024

@author: 37092
"""

import pandas as pd
import numpy as np

sm_prepocessed = pd.read_csv(r"D:\conference\data_platform\after_preprocess_data.csv")
label = pd.read_csv(r"D:\conference\data_platform\label.csv")

thief = sm_prepocessed[sm_prepocessed.index.isin(label[label.flag ==1].index)]
normal = sm_prepocessed[~sm_prepocessed.index.isin(label[label.flag ==1].index)]

zero_rows = thief[thief.eq(0).all(axis=1)]
nonzero_rows = thief[~thief.eq(0).all(axis=1)]


def has_continuous_zeros(row, threshold=30):
    consecutive_zeros = 0
    for value in row:
        if value == 0:
            consecutive_zeros += 1
            if consecutive_zeros > threshold:
                return True
        else:
            consecutive_zeros = 0
    return False

selected_rows = nonzero_rows[nonzero_rows.apply(has_continuous_zeros, axis=1)]



####################################################################################################
zero_rows_normal = normal[normal.eq(0).all(axis=1)]
nonzero_rows_normal = normal[~normal.eq(0).all(axis=1)]


selected_rows_normal = nonzero_rows_normal[nonzero_rows_normal.apply(has_continuous_zeros, axis=1)]


import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Determine the color limits based on the combined data of both matrices
vmin = min(nonzero_rows.values.min(), nonzero_rows_normal.values.min())
vmax = max(nonzero_rows.values.max(), nonzero_rows_normal.values.max())

# Plot the first heatmap
sns.heatmap(nonzero_rows, ax=ax1, vmin=vmin, vmax=vmax, cmap="viridis")
ax1.set_title('Heatmap of nonzero_rows')

# Plot the second heatmap
sns.heatmap(nonzero_rows_normal, ax=ax2, vmin=vmin, vmax=vmax, cmap="viridis")
ax2.set_title('Heatmap of nonzero_rows_normal')

# Display the plot
plt.tight_layout()
plt.show()



original_data = pd.read_csv(r"D:\conference\data_platform\ElectricityTheftDetection-master\data\data.csv")

original_data_theft = original_data[original_data.FLAG == 1]
original_data_normal = original_data[original_data.FLAG == 0]

# original_data_theft_nan = original_data_theft[original_data_theft.iloc[:, 2:].isnull().all(axis=1)]
# original_data_theft_zero = original_data_theft[original_data_theft.iloc[:, 2:].eq(0).all(axis=1)]
# original_data_normal_nan = original_data_normal[original_data_normal.iloc[:, 2:].isnull().all(axis=1)]
# original_data_normal_zero = original_data_normal[original_data_normal.iloc[:, 2:].eq(0).all(axis=1)]

theft_condition_nan_or_zero = original_data_theft.iloc[:, 2:].isnull() | (original_data_theft.iloc[:, 2:] == 0)
original_data_theft_nan_or_zero = original_data_theft[theft_condition_nan_or_zero.all(axis=1)]

normal_condition_nan_or_zero = original_data_normal.iloc[:, 2:].isnull() | (original_data_normal.iloc[:, 2:] == 0)
original_data_normal_nan_or_zero = original_data_normal[normal_condition_nan_or_zero.all(axis=1)]

usable_theft = original_data_theft[~theft_condition_nan_or_zero.all(axis=1)]
usable_normal = original_data_normal[~normal_condition_nan_or_zero.all(axis=1)]

usable_theft = usable_theft.drop(columns=['CONS_NO', 'FLAG'])
usable_normal = usable_normal.drop(columns=['CONS_NO', 'FLAG'])


usable_theft = usable_theft.T
usable_normal = usable_normal.T

usable_theft.index = pd.to_datetime(usable_theft.index, format='%Y/%m/%d')
usable_normal.index = pd.to_datetime(usable_normal.index, format='%Y/%m/%d')

usable_theft = usable_theft.sort_index()
usable_normal = usable_normal.sort_index()

usable_theft = usable_theft.T
usable_normal = usable_normal.T



usable_theft = usable_theft.drop(index=2197)
usable_theft = usable_theft.drop(index=2941)
usable_theft = usable_theft.drop(index=3403)
usable_theft = usable_theft.drop(index=2845)
usable_theft = usable_theft.drop(index=1359)
usable_theft = usable_theft.drop(index=3539)


# Assuming df is your DataFrame
value_to_find =7854.0# 26857.59#27180.0#57062.68#92713.9#466714.4#514991.78

# Find the location of the value
result = usable_theft[usable_theft == value_to_find].stack()

# If result is not empty, print the location
if not result.empty:
    print("Location(s) of the value", value_to_find, ":")
    for idx, val in result.index:
        print("Row:", idx, ", Column:", val)
else:
    print("Value", value_to_find, "not found in the DataFrame.")


usable_normal = usable_normal.drop(index=7435)
usable_normal = usable_normal.drop(index=24281)
usable_normal = usable_normal.drop(index=8217)
usable_normal = usable_normal.drop(index=29989)
usable_normal = usable_normal.drop(index=31384)
usable_normal = usable_normal.drop(index=11490)
usable_normal = usable_normal.drop(index=31054)
usable_normal = usable_normal.drop(index=32332)
usable_normal = usable_normal.drop(index=28438)
usable_normal = usable_normal.drop(index=41229)
usable_normal = usable_normal.drop(index=42346)
usable_normal = usable_normal.drop(index=5072)
usable_normal = usable_normal.drop(index=5514)


# Assuming df is your DataFrame
value_to_find =13304.19#13442.31#14408.43#14425.34 #16331.05#17812.3# 18526.38#29910.91#53532.0#554535.44#640000.0#653502.79#800003.32

# Find the location of the value
result = usable_normal[usable_normal == value_to_find].stack()

# If result is not empty, print the location
if not result.empty:
    print("Location(s) of the value", value_to_find, ":")
    for idx, val in result.index:
        print("Row:", idx, ", Column:", val)
else:
    print("Value", value_to_find, "not found in the DataFrame.")


usable_theft_backup = usable_theft.copy()
usable_normal_backup = usable_normal.copy()

# # Replace zeros with NaNs to exclude them from mean and standard deviation calculations
# usable_theft.replace(0, np.nan, inplace=True)
# usable_normal.replace(0, np.nan, inplace=True)

theft_mean = usable_theft.mean()
theft_std_dev = usable_theft.std()
normal_mean = usable_normal.mean()
normal_std_dev = usable_normal.std()

usable_theft = usable_theft[~(np.abs((usable_theft - theft_mean) / theft_std_dev) > 3).any(axis=1)]
usable_normal = usable_normal[~(np.abs((usable_normal - normal_mean) / normal_std_dev) > 3).any(axis=1)]





import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Calculate vmin and vmax considering NaN values
vmin = min(np.nanmin(usable_theft.values), np.nanmin(usable_normal.values))
vmax = max(np.nanmax(usable_theft.values), np.nanmax(usable_normal.values))

# Plot the first heatmap
sns.heatmap(usable_theft, ax=ax1, vmin=vmin, vmax=vmax, cmap="viridis")
ax1.set_title('Heatmap of usable_theft')

# Plot the second heatmap
sns.heatmap(usable_normal, ax=ax2, vmin=vmin, vmax=vmax, cmap="viridis")
ax2.set_title('Heatmap of usable_normal')

# Display the plot
plt.tight_layout()
plt.show()




