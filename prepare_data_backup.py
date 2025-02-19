import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from data_add_noise.attack_func import *

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
def has_continuous_nans(row, threshold=2):
    consecutive_nans = 0
    for value in row:
        if pd.isna(value):
            consecutive_nans += 1
            if consecutive_nans > threshold:
                return True
        else:
            consecutive_nans = 0
    return False

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

usable_normal_2016_linear_interpolated = usable_normal_2016.interpolate(method='linear', axis=1)
usable_normal_2016_linear_interpolated = usable_normal_2016_linear_interpolated[~usable_normal_2016_linear_interpolated.isnull().any(axis=1)]

usable_normal_2016_max_min_diff = usable_normal_2016_linear_interpolated.max(axis=1) - usable_normal_2016_linear_interpolated.min(axis=1)

# z=usable_normal_2016_linear_interpolated[usable_normal_2016_linear_interpolated.index.isin(usable_normal_2016_max_min_diff[usable_normal_2016_max_min_diff>1000].index)]
# z=z.T

############# 假设严格一些：3sigma原则
# 计算每行的均值和标准差：row_mean, row_std
usable_normal_2016_linear_interpolated['row_mean'] = usable_normal_2016_linear_interpolated.mean(axis=1)
usable_normal_2016_linear_interpolated['row_std'] = usable_normal_2016_linear_interpolated.std(axis=1)
# 计算每行的下界
usable_normal_2016_linear_interpolated['lower_bound'] = usable_normal_2016_linear_interpolated['row_mean'] - 3 * usable_normal_2016_linear_interpolated['row_std']
# 计算每行的上界
usable_normal_2016_linear_interpolated['upper_bound'] = usable_normal_2016_linear_interpolated['row_mean'] + 3 * usable_normal_2016_linear_interpolated['row_std']
# 过滤掉不在3σ范围内的行
df_no_outliers1 = usable_normal_2016_linear_interpolated.apply(lambda row: any((row[:len(usable_normal_2016_linear_interpolated.columns)-4] < row['lower_bound']) | 
                                                                               (row[:len(usable_normal_2016_linear_interpolated.columns)-4] > row['upper_bound'])), axis=1)
# 删除辅助列
usable_normal_check_mask = usable_normal_2016_linear_interpolated[~df_no_outliers1]

usable_normal_2016_3sigma_back = usable_normal_check_mask.copy()
usable_normal_2016_3sigma = usable_normal_check_mask.drop(columns=['row_mean', 'row_std', 'lower_bound', 'upper_bound'])

# 为每个用户生成两个0到1之间的随机数
# start_rates = np.random.rand(len(usable_normal_2016_3sigma))
# end_rates = np.random.rand(len(usable_normal_2016_3sigma))
start_rates = np.random.uniform(0, 0.3, len(usable_normal_2016_3sigma))
end_rates = np.random.uniform(0.7, 1, len(usable_normal_2016_3sigma))
# 确保 end_rates 对应的位置上的数字比 start_rates 上的大
mask = end_rates < start_rates
start_rates[mask], end_rates[mask] = end_rates[mask], start_rates[mask]

# 创建一个新的DataFrame来存储攻击的结果
attack_df = usable_normal_2016_3sigma.copy()
# 创建一个新的DataFrame来存储标签
label_df = pd.DataFrame(0, index=usable_normal_2016_3sigma.index, columns=usable_normal_2016_3sigma.columns)

# group_normal_2016 = group_normal_2016_selected_rows.copy()

print(f"Time length: {len(usable_normal_2016_3sigma.columns)}")

'''
def type1_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
def type2_attack(data, start_point, duration, sigma):
def type3_attack(data, start_point, duration, gamma):
def type4_attack(data, start_point, duration):
def type5_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
def type6_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
'''

# 对每个用户施加攻击
id = 3
err=[]
for i in range(len(usable_normal_2016_3sigma)):
    # 计算开始索引和结束索引
    start_index = int(start_rates[i] * len(usable_normal_2016_3sigma.columns))
    end_index = int(end_rates[i] * len(usable_normal_2016_3sigma.columns))

    # 计算持续时间
    duration = end_index - start_index

    # 将原始的DataFrame转换为数组
    data_array = np.expand_dims(usable_normal_2016_3sigma.iloc[i].values, axis=0)
    data_array_copy = usable_normal_2016_3sigma.copy().iloc[i].values

    # 在数组上施加攻击
    switcher = {
        1: type1_attack,
        2: type2_attack,
        3: type3_attack,
        4: type4_attack,
        5: type5_attack,
        6: type6_attack
    }
    attack_func = switcher.get(id)
    if attack_func:
        attack_result = attack_func(np.expand_dims(data_array, axis=1), start_index, duration)
    else:
        attack_result = None

    # if id == 1:
    #     attack_result = type1_attack(np.expand_dims(data_array, axis=1), start_index, duration)
    # elif id == 2:
    #     attack_result = type2_attack(np.expand_dims(data_array, axis=1), start_index, duration)
    # elif id == 3:
    #     attack_result = type3_attack(np.expand_dims(data_array, axis=1), start_index, duration)
    # elif id == 4:
    #     attack_result = type4_attack(np.expand_dims(data_array, axis=1), start_index, duration)
    # elif id == 5:
    #     attack_result = type5_attack(np.expand_dims(data_array, axis=1), start_index, duration)
    # elif id == 6:
    #     attack_result = type6_attack(np.expand_dims(data_array, axis=1), start_index, duration)

    print( np.sum(data_array_copy- attack_result[0,0,:]))
    err.append(np.sum(data_array_copy- attack_result[0,0,:]))

    if np.sum(data_array_copy- attack_result[0,0,:]) > 100:###一年窃电100kwh以上
        # 将攻击结果存储到attack_df中
        attack_df.iloc[i, :] = attack_result.squeeze()
    
        # 更新标签
        label_df.iloc[i, start_index:end_index] = 1



zy = label_df[(label_df != 0).any(axis=1)] 
zx = attack_df[attack_df.index.isin(zy.index)] 

# 将攻击的结果和标签存储为两个CSV文件
zx.to_csv(f'zx{id}.csv', index=False)
zy.to_csv(f'zy{id}.csv', index=False)
zy.to_csv(f'./data_prepared/zy{id}.csv', index=False)


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
df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized.T).T, columns=zx.columns)

df_normalized.to_csv(f'zx{id}_normalized.csv', index=False)
df_normalized.to_csv(f'./data_prepared/zx{id}_normalized.csv', index=False)



# df2_normalized = df2.copy()
# df2_normalized = pd.DataFrame(scaler.fit_transform(df2_normalized.T).T, columns=df2_normalized.columns)
# df2_normalized.to_csv('Normal3_normalized.csv', index=False)
# df2_normalized.to_csv('./data_prepared/Normal3_normalized.csv', index=False)

# label_df2 = pd.DataFrame(0, index=df2.index, columns=df2.columns)
# label_df2.to_csv('Normal3_normalized_label.csv', index=False)
# label_df2.to_csv('./data_prepared/Normal3_normalized_label.csv', index=False)