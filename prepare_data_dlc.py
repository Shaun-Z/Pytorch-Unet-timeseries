# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

import sys

# Add root path to the system path
from data_add_noise.attack_func import *

# %%
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

def get_args():
    parser = argparse.ArgumentParser(description='Prepare the data for the attack')
    parser.add_argument('--attack_id', '-a', type=int, default=1, help='Attack ID')
    parser.add_argument('--data_name', type=str, default='DLC', help='Name of the dataset')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    attack_id = args.attack_id
    path_to_data = f'./data/{args.data_name}_data'
    os.makedirs(f'{path_to_data}/data_prepared_{attack_id}', exist_ok=True)
    # %%
    original_data = pd.read_csv(f"{path_to_data}/dlc_after_3sigma.csv")
    # %%
    original_data_normal = original_data

    normal_condition_nan_or_zero = original_data_normal.iloc[:, 2:].isnull() | (original_data_normal.iloc[:, 2:] == 0)
    original_data_normal_nan_or_zero = original_data_normal[normal_condition_nan_or_zero.all(axis=1)]

    usable_normal = original_data_normal[~normal_condition_nan_or_zero.all(axis=1)]

    usable_normal = usable_normal.drop(columns=['BADGE_NBR'])

    # %%
    usable_normal = usable_normal.T
    usable_normal.index = pd.to_datetime(usable_normal.index, format='%Y-%m-%d %H:%M:%S')
    usable_normal = usable_normal.sort_index()

    # %%
    usable_normal = usable_normal.T

    # %%
    usable_normal = usable_normal[~usable_normal.apply(has_continuous_zeros, axis=1)]

    # %%
    usable_normal = usable_normal[~usable_normal.apply(has_continuous_nans, axis=1)] #### can relax to larger continous value

    # %%
    # usable_normal_linear_interpolated = usable_normal.interpolate(method='linear', axis=1)
    # usable_normal_linear_interpolated = usable_normal_linear_interpolated[~usable_normal_linear_interpolated.isnull().any(axis=1)]

    # usable_normal_max_min_diff = usable_normal_linear_interpolated.max(axis=1) - usable_normal_linear_interpolated.min(axis=1)

    # %%
    # ############# 假设严格一些：3sigma原则
    # # 计算每行的均值和标准差：row_mean, row_std
    # usable_normal_linear_interpolated['row_mean'] = usable_normal_linear_interpolated.mean(axis=1)
    # usable_normal_linear_interpolated['row_std'] = usable_normal_linear_interpolated.std(axis=1)
    # # 计算每行的下界
    # usable_normal_linear_interpolated['lower_bound'] = usable_normal_linear_interpolated['row_mean'] - 3 * usable_normal_linear_interpolated['row_std']
    # # 计算每行的上界
    # usable_normal_linear_interpolated['upper_bound'] = usable_normal_linear_interpolated['row_mean'] + 3 * usable_normal_linear_interpolated['row_std']
    # # 过滤掉不在3σ范围内的行
    # df_no_outliers1 = usable_normal_linear_interpolated.apply(lambda row: any((row[:len(usable_normal_linear_interpolated.columns)-4] < row['lower_bound']) | 
    #                                                                                (row[:len(usable_normal_linear_interpolated.columns)-4] > row['upper_bound'])), axis=1)
    # # 删除辅助列
    # usable_normal_check_mask = usable_normal_linear_interpolated[~df_no_outliers1]

    # usable_normal_3sigma_back = usable_normal_check_mask.copy()
    # usable_normal_3sigma = usable_normal_check_mask.drop(columns=['row_mean', 'row_std', 'lower_bound', 'upper_bound'])

    # Shuffle the DataFrame rows
    df_shuffled = usable_normal.sample(frac=1, random_state=42).reset_index(drop=True)
    # df1 = df_shuffled.iloc[:df_shuffled.shape[0]//2]
    df1 = df_shuffled.iloc[:df_shuffled.shape[0]//4*3]
    df2 = df_shuffled.iloc[df_shuffled.shape[0]//4*3:]

    df_shuffled1 = df_shuffled.iloc[:df_shuffled.shape[0]//2]
    df_shuffled2 = df_shuffled.iloc[df_shuffled.shape[0]//2:]

    df_ones = pd.DataFrame(1, index=df_shuffled1.index, columns=df_shuffled1.columns)
    df_zeros = pd.DataFrame(0, index=df_shuffled2.index, columns=df_shuffled2.columns)

    df_data = pd.concat([df_shuffled1, df_shuffled2], ignore_index=True)
    # df_data.to_csv('test_data.csv', index=False)
    df_combined = pd.concat([df_ones, df_zeros], ignore_index=True)
    # df_combined.to_csv('test_label.csv', index=False)

    # %%
    df_shuffled.shape

    # %%
    # 为每个用户生成两个0到1之间的随机数
    start_rates = np.random.rand(len(df_shuffled))
    end_rates = np.random.rand(len(df_shuffled))
    # start_rates = np.random.uniform(0, 0.3, len(df1))
    # end_rates = np.random.uniform(0.7, 1, len(df1))
    # 确保 end_rates 对应的位置上的数字比 start_rates 上的大
    mask = end_rates < start_rates
    start_rates[mask], end_rates[mask] = end_rates[mask], start_rates[mask]

    # 创建一个新的DataFrame来存储攻击的结果
    attack_df = df1.copy()
    # 创建一个新的DataFrame来存储标签
    label_df = pd.DataFrame(0, index=df1.index, columns=df1.columns)

    # group_normal = group_normal_selected_rows.copy()

    print(f"Time length: {len(df1.columns)}")

    # %%
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
        switcher = {
            1: type1_attack,
            2: type2_attack,
            3: type3_attack,
            4: type4_attack,
            5: type5_attack,
            6: type6_attack
        }
        attack_func = switcher.get(attack_id)
        if attack_func:
            attack_result = attack_func(np.expand_dims(data_array, axis=1), start_index, duration)
        else:
            attack_result = None

        # if attack_id == 1:
        #     attack_result = type1_attack(np.expand_dims(data_array, axis=1), start_index, duration)
        # elif attack_id == 2:
        #     attack_result = type2_attack(np.expand_dims(data_array, axis=1), start_index, duration)
        # elif attack_id == 3:
        #     attack_result = type3_attack(np.expand_dims(data_array, axis=1), start_index, duration)
        # elif attack_id == 4:
        #     attack_result = type4_attack(np.expand_dims(data_array, axis=1), start_index, duration)
        # elif attack_id == 5:
        #     attack_result = type5_attack(np.expand_dims(data_array, axis=1), start_index, duration)
        # elif attack_id == 6:
        #     attack_result = type6_attack(np.expand_dims(data_array, axis=1), start_index, duration)

        print( np.sum(data_array_copy- attack_result[0,0,:]))
        err.append(np.sum(data_array_copy- attack_result[0,0,:]))

        # if np.sum(data_array_copy- attack_result[0,0,:]) > 100:###一年窃电100kwh以上
        #     # 将攻击结果存储到attack_df中
        #     attack_df.iloc[i, :] = attack_result.squeeze()
        
        #     # 更新标签
        #     label_df.iloc[i, start_index:end_index] = 1

        if np.sum(data_array_copy- attack_result[0,0,:]) > np.sum(data_array_copy)*0.05 and np.sum(data_array_copy- attack_result[0,0,:]) > 10: ###一年窃电10%以上
            # 将攻击结果存储到attack_df中
            attack_df.iloc[i, :] = attack_result.squeeze()
        
            # 更新标签
            label_df.iloc[i, start_index:end_index] = 1

    # %%
    zy = label_df[(label_df != 0).any(axis=1)]
    zx = attack_df[attack_df.index.isin(zy.index)]

    # %%
    # 将攻击的结果和标签存储为两个CSV文件
    os.makedirs(f'./{path_to_data}/data_prepared_{attack_id}', exist_ok=True)
    # zx.to_csv(f'zx{attack_id}.csv', index=False)
    # zy.to_csv(f'zy{attack_id}.csv', index=False)
    zy.to_csv(f'./{path_to_data}/data_prepared_{attack_id}/zy{attack_id}.csv', index=False)

    # %%
    from sklearn.preprocessing import MinMaxScaler
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    # Apply MinMaxScaler to each row
    df_normalized = zx.copy()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized.T).T, columns=zx.columns)

    # df_normalized.to_csv(f'zx{attack_id}_normalized.csv', index=False)
    df_normalized.to_csv(f'./{path_to_data}/data_prepared_{attack_id}/zx{attack_id}_normalized.csv', index=False)


    df2_normalized = df2.copy()
    # df2_normalized = df1.copy()
    df2_normalized = pd.DataFrame(scaler.fit_transform(df2_normalized.T).T, columns=df2_normalized.columns)
    # df2_normalized.to_csv('Normal3_normalized.csv', index=False)
    df2_normalized.to_csv(f'./{path_to_data}/data_prepared_{attack_id}/Normal3_normalized.csv', index=False)

    label_df2 = pd.DataFrame(0, index=df2.index, columns=df2.columns)
    # label_df2.to_csv('Normal3_normalized_label.csv', index=False)
    label_df2.to_csv(f'./{path_to_data}/data_prepared_{attack_id}/Normal3_normalized_label.csv', index=False)

    label_df2.iloc[:, -12:] = [0,1,1,1,0,1,1,1,0,1,1,1]
    # label_df2.to_csv('Normal3_normalized_pseudolabel.csv', index=False)
    label_df2.to_csv(f'./{path_to_data}/data_prepared_{attack_id}/Normal3_normalized_pseudolabel.csv', index=False)

    zx3_normalized = pd.read_csv(f'./{path_to_data}/data_prepared_{attack_id}/zx{attack_id}_normalized.csv') 
    normal3_normalized = pd.read_csv(f'./{path_to_data}/data_prepared_{attack_id}/Normal3_normalized.csv')

    zy3 = pd.read_csv(f'./{path_to_data}/data_prepared_{attack_id}/zy{attack_id}.csv')
    normal3_normalized_label = pd.read_csv(f'./{path_to_data}/data_prepared_{attack_id}/Normal3_normalized_label.csv')
    normal3_normalized_label.columns = zy3.columns

    combined_dfx = pd.concat([zx3_normalized, normal3_normalized], ignore_index=True)#
    combined_dfy = pd.concat([zy3, normal3_normalized_label], ignore_index=True)#


    combined_dfx.to_csv(f'./{path_to_data}/data_prepared_{attack_id}/combined_dfx.csv', index=False)
    combined_dfy.to_csv(f'./{path_to_data}/data_prepared_{attack_id}/combined_dfy.csv', index=False)

    # -------

    normal3_normalized_pseudolabel = pd.read_csv(f'./{path_to_data}/data_prepared_{attack_id}/Normal3_normalized_pseudolabel.csv')
    normal3_normalized_pseudolabel.columns = zy3.columns
    combined_dfy_pseudo = pd.concat([zy3, normal3_normalized_pseudolabel], ignore_index=True)#
    combined_dfy_pseudo.to_csv(f'./{path_to_data}/data_prepared_{attack_id}/combined_dfy_pseudo.csv', index=False)
