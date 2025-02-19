# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 10:42:35 2024

@author: 37092
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# from data_add_noise.attack_func import *

'''
定义两个函数：
has_continuous_zeros(row, threshold=30)：检查一行数据中是否有连续超过 threshold 个小于等于 0.17 的值。如果有，返回 True，否则返回 False。
'''
def has_continuous_zeros(row, threshold=30):
    consecutive_zeros = 0
    for value in row:
        if value <= 0.17:
            consecutive_zeros += 1
            if consecutive_zeros > threshold:
                return True
        else:
            consecutive_zeros = 0
    return False
'''
has_continuous_nans(row, threshold=2)：检查一行数据中是否有连续超过 threshold 个 NaN 值。如果有，返回 True，否则返回 False。
'''
def has_continuous_nans(row, threshold=3):
    consecutive_nans = 0
    for value in row:
        if pd.isna(value):
            consecutive_nans += 1
            if consecutive_nans > threshold:
                return True
        else:
            consecutive_nans = 0
    return False

def type3_attack(data, start_point, duration, gamma):
    end_point = start_point + duration
    data[:, :, start_point:end_point] *= (1-gamma)
    data = np.maximum(data, 0)  # Ensure no negative values
    return data


original_data = pd.read_csv("./data/SGCC_data/data.csv")

original_data_theft = original_data[original_data.FLAG == 1]
original_data_normal = original_data[original_data.FLAG == 0]

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


usable_theft_2016 = usable_theft[usable_theft.index.year == 2016]
usable_normal_2016 = usable_normal[usable_normal.index.year == 2016]
usable_theft_2016 = usable_theft_2016.T
usable_normal_2016 = usable_normal_2016.T
print(usable_normal_2016.shape)


usable_theft = usable_theft.T
usable_normal = usable_normal.T
###################### preprocessing for 2016 normal data

usable_normal_2016 = usable_normal_2016[~usable_normal_2016.apply(has_continuous_zeros, axis=1)]
print(usable_normal_2016.shape)

usable_normal_2016 = usable_normal_2016[~usable_normal_2016.apply(has_continuous_nans, axis=1)] #### can relax to larger continous value
print(usable_normal_2016.shape)



# Transpose the DataFrame so that days become rows (needed for rolling)
df_transposed = usable_normal_2016.T
# Apply the rolling window (e.g., 7-day) and calculate the rolling mean
rolling_mean = df_transposed.rolling(window=7, min_periods=1).mean().T

# Calculate the rolling standard deviation
rolling_std = df_transposed.rolling(window=7, min_periods=1).std().T

# # Identify outliers: Points that are more than `threshold` standard deviations from the rolling mean
# outliers = (usable_normal_2016 - rolling_mean).abs() > (3 * rolling_std)
# # Replace outliers with -99
# df_with_replaced_outliers = usable_normal_2016.mask(outliers, -99)
# # Remove rows with any -99 values
# df_cleaned = df_with_replaced_outliers[~df_with_replaced_outliers.eq(-99).any(axis=1)]
 
# outliers = (usable_normal_2016 - rolling_mean).abs() > (2 * rolling_std)
# # Replace outliers with -99
# df_with_replaced_outliers = usable_normal_2016.mask(outliers, -99)
# # Remove rows with any -99 values
# df_cleaned = df_with_replaced_outliers[~df_with_replaced_outliers.eq(-99).any(axis=1)]
 
# outliers = (usable_normal_2016 - rolling_mean).abs() > (2.5 * rolling_std)
# # Replace outliers with -99
# df_with_replaced_outliers = usable_normal_2016.mask(outliers, -99)
# # Remove rows with any -99 values
# df_cleaned = df_with_replaced_outliers[~df_with_replaced_outliers.eq(-99).any(axis=1)]

# outliers = (usable_normal_2016 - rolling_mean).abs() > (2.3 * rolling_std)
# # Replace outliers with -99
# df_with_replaced_outliers = usable_normal_2016.mask(outliers, -99)
# # Remove rows with any -99 values
# df_cleaned = df_with_replaced_outliers[~df_with_replaced_outliers.eq(-99).any(axis=1)]
 
outliers = (usable_normal_2016 - rolling_mean).abs() > (2.2 * rolling_std)
# Replace outliers with -99
df_with_replaced_outliers = usable_normal_2016.mask(outliers, -99)
# Remove rows with any -99 values
df_cleaned = df_with_replaced_outliers[~df_with_replaced_outliers.eq(-99).any(axis=1)]
plt.figure()
sns.heatmap(df_cleaned)

# from sklearn.preprocessing import MinMaxScaler
# # Initialize the MinMaxScaler
# scaler = MinMaxScaler()
# # Apply MinMaxScaler to each row
# df_normalized = df_cleaned.copy()
# df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized.T).T, columns=df_cleaned.columns)
# plt.figure()
# sns.heatmap(df_normalized)




df_cleaned_interpolated = df_cleaned.interpolate(method='linear', axis=1)
df_cleaned_interpolated = df_cleaned_interpolated[~df_cleaned_interpolated.isnull().any(axis=1)]

df_cleaned_max_min_diff = df_cleaned_interpolated.max(axis=1) - df_cleaned_interpolated.min(axis=1)


# Shuffle the DataFrame rows
df_shuffled = df_cleaned_interpolated.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the DataFrame into two equal parts
df1 = df_shuffled.iloc[:6000]
df2 = df_shuffled.iloc[6000:]



####################################################################
# 为每个用户生成两个0到1之间的随机数
# start_rates = np.random.rand(len(df1))
# end_rates = np.random.rand(len(df1))
start_rates = np.random.uniform(0, 0.3, len(df1))
end_rates = np.random.uniform(0.7, 1, len(df1))

# 确保 end_rates 对应的位置上的数字比 start_rates 上的大
mask = end_rates < start_rates
start_rates[mask], end_rates[mask] = end_rates[mask], start_rates[mask]

# 创建一个新的DataFrame来存储攻击的结果
attack_df = df1.copy()
# 创建一个新的DataFrame来存储标签
label_df = pd.DataFrame(0, index=df1.index, columns=df1.columns)

print(f"Time length: {len(df1.columns)}")

'''
def type1_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
def type2_attack(data, start_point, duration, sigma):
def type3_attack(data, start_point, duration, gamma):
def type4_attack(data, start_point, duration):
def type5_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
def type6_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
'''

# 对每个用户施加攻击
err=[]
for i in range(len(df1)):
    # 计算开始索引和结束索引
    start_index = int(start_rates[i] * len(df1.columns))
    end_index = int(end_rates[i] * len(df1.columns))-12

    # 计算持续时间
    duration = end_index - start_index

    # 将原始的DataFrame转换为数组
    data_array = np.expand_dims(df1.iloc[i].values, axis=0)
    data_array_copy = df1.copy().iloc[i].values

    # 在数组上施加攻击
    # attack_result = type1_attack(np.expand_dims(data_array, axis=1), start_index, duration)

    # alpha, attack_result = type1_attack(np.expand_dims(data_array, axis=1), start_index, duration)
    # print(f"{alpha}, {start_index} - {end_index} : {duration}")

    # attack_result = type2_attack(np.expand_dims(data_array, axis=1), start_index, duration, 0.1)
    attack_result = type3_attack(np.expand_dims(data_array, axis=1), start_index, duration, 0.1)

    print( np.sum(data_array_copy- attack_result[0,0,:]))
    err.append(np.sum(data_array_copy- attack_result[0,0,:]))

    # TODO: Other standards for determining if an attack is successful 按照全年用电量的峰值的百分比来判断。或总用电量的百分比。
    if np.sum(data_array_copy- attack_result[0,0,:]) > 100:###一年窃电100kwh以上
        # 将攻击结果存储到attack_df中
        attack_df.iloc[i, :] = attack_result.squeeze()
    
        # 更新标签
        label_df.iloc[i, start_index:end_index] = 1

    # if np.sum(data_array_copy- attack_result[0,0,:]) > np.sum(data_array_copy)*0.05: ###一年窃电10%以上
    #     # 将攻击结果存储到attack_df中
    #     attack_df.iloc[i, :] = attack_result.squeeze()
    
    #     # 更新标签
    #     label_df.iloc[i, start_index:end_index] = 1

zy = label_df[(label_df != 0).any(axis=1)] 
zx = attack_df[attack_df.index.isin(zy.index)] 

# 将攻击的结果和标签存储为两个CSV文件
# zx.to_csv('zx3.csv', index=False)
zy.to_csv('Attack3_normalized_label.csv', index=False)
zy.to_csv('./data_prepared/Attack3_normalized_label.csv', index=False)


# # Create a figure with two subplots
# fig, ax1 = plt.subplots()

# # Calculate vmin and vmax considering NaN values
# vmin = np.nanmin(usable_normal_2016.values)
# vmax = np.nanmax(usable_normal_2016.values)


# # Plot the second heatmap
# sns.heatmap(usable_normal_2016, ax=ax1, vmin=vmin, vmax=vmax, cmap="viridis")
# ax1.set_title('Heatmap of usable_normal_2016')

# # Display the plot
# plt.tight_layout()
# plt.show()

from sklearn.preprocessing import MinMaxScaler
# Initialize the MinMaxScaler
scaler = MinMaxScaler()
# Apply MinMaxScaler to each row
df_normalized = zx.copy()
df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized.T).T, columns=df_normalized.columns)

df_normalized.to_csv('Attack3_normalized.csv', index=False)
df_normalized.to_csv('./data_prepared/Attack3_normalized.csv', index=False)


df2_normalized = df2.copy()
df2_normalized = pd.DataFrame(scaler.fit_transform(df2_normalized.T).T, columns=df2_normalized.columns)
df2_normalized.to_csv('Normal3_normalized.csv', index=False)
df2_normalized.to_csv('./data_prepared/Normal3_normalized.csv', index=False)

label_df2 = pd.DataFrame(0, index=df2.index, columns=df2.columns)
label_df2.to_csv('Normal3_normalized_label.csv', index=False)
label_df2.to_csv('./data_prepared/Normal3_normalized_label.csv', index=False)

# # Set the last two columns of each row to 1
# label_df2.iloc[:, -10:-1] = 1

# label_df2.to_csv('Normal3_normalized_pseudolabel.csv', index=False)
# label_df2.to_csv('./data_prepared/Normal3_normalized_pseudolabel.csv', index=False)


# label_df3 = pd.DataFrame(0, index=df2.index, columns=df2.columns)
# # Set the last two columns of each row to 1
# # label_df3.iloc[:, -4:-1] = 1
# label_df3.iloc[:, -12:] = [0,1,1,0,0,1,1,0,0,1,1,0]
# label_df3.to_csv('Normal3_normalized_pseudolabel1.csv', index=False)
# label_df3.to_csv('./data_prepared/Normal3_normalized_pseudolabel1.csv', index=False)


# label_df4 = pd.DataFrame(0, index=df2.index, columns=df2.columns)
# # Set the last two columns of each row to 1
# # label_df4.iloc[:, -2:-1] = 1
# label_df4.iloc[:, -12:] = [0,1,1,1,0,1,1,1,0,1,1,1]
# label_df4.to_csv('Normal3_normalized_pseudolabel2.csv', index=False)
# label_df4.to_csv('./data_prepared/Normal3_normalized_pseudolabel2.csv', index=False)

label_df2.iloc[:, -12:] = [0,1,1,1,0,1,1,1,0,1,1,1]
label_df2.to_csv('Normal3_normalized_pseudolabel.csv', index=False)
label_df2.to_csv('./data_prepared/Normal3_normalized_pseudolabel.csv', index=False)

# -------

# zx3_normalized = pd.read_csv(f'./data_prepared/Attack3_normalized.csv') 
zx3_normalized = pd.read_csv(f'./data_prepared/zx3_normalized.csv') 
normal3_normalized = pd.read_csv(f'./data_prepared/Normal3_normalized.csv')

# -------
normal3_normalized.columns = zx3_normalized.columns

# zy3 = pd.read_csv(f'./data_prepared/Attack3_normalized_label.csv') 
zy3 = pd.read_csv(f'./data_prepared/zy3.csv') 
normal3_normalized_label = pd.read_csv(f'./data_prepared/Normal3_normalized_label.csv')
normal3_normalized_label.columns = zy3.columns

combined_dfx = pd.concat([zx3_normalized, normal3_normalized], ignore_index=True)#
combined_dfy = pd.concat([zy3, normal3_normalized_label], ignore_index=True)#


combined_dfx.to_csv('./data_prepared/combined_dfx.csv', index=False)
combined_dfy.to_csv('./data_prepared/combined_dfy.csv', index=False)

# -------

normal3_normalized_pseudolabel = pd.read_csv(f'./data_prepared/Normal3_normalized_pseudolabel.csv')
normal3_normalized_pseudolabel.columns = zy3.columns
combined_dfy_sudo = pd.concat([zy3, normal3_normalized_pseudolabel], ignore_index=True)#
combined_dfy_sudo.to_csv('./data_prepared/combined_dfy_pseudo.csv', index=False)