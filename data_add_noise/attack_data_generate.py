# %%
from attack_func import *
from pathlib import Path
import pandas as pd

# %%
datapath = Path('data/SGCC_data')
df = pd.read_csv(datapath/"data.csv")
df = df.set_index('FLAG').drop('CONS_NO', axis=1)
df.columns = pd.to_datetime(df.columns)
df = df.sort_index(axis=1)
df.shape

# %%
groups = df.groupby(df.index, sort=False)
group_normal = groups.get_group(0)
group_attack = groups.get_group(1)
group_normal.shape, group_attack.shape

# %%
threshold = 5  # 设置阈值，根据实际情况调整
group_normal = group_normal.dropna(thresh=df.shape[1]-threshold)
group_normal.shape

# %%
group_normal_2016 = group_normal.loc[:, (group_normal.columns.year == 2016)].reset_index(drop=True).fillna(0)
group_normal_2016

# %%
# 为每个用户生成两个0到1之间的随机数
start_rates = np.random.rand(len(group_normal_2016))
end_rates = np.random.rand(len(group_normal_2016))
# 确保 end_rates 对应的位置上的数字比 start_rates 上的大
mask = end_rates < start_rates
start_rates[mask], end_rates[mask] = end_rates[mask], start_rates[mask]

# %%
# 创建一个新的DataFrame来存储攻击的结果
attack_df = group_normal_2016.copy()
# 创建一个新的DataFrame来存储标签
label_df = pd.DataFrame(0, index=group_normal_2016.index, columns=group_normal_2016.columns)

# %%
print(f"Time length: {len(group_normal_2016.columns)}")
# 对每个用户施加攻击
for i in range(len(group_normal_2016)):
    # 计算开始索引和结束索引
    start_index = int(start_rates[i] * len(group_normal_2016.columns))
    end_index = int(end_rates[i] * len(group_normal_2016.columns))

    # 计算持续时间
    duration = end_index - start_index

    # 将原始的DataFrame转换为数组
    data_array = np.expand_dims(group_normal_2016.iloc[i].values, axis=0)

    # 在数组上施加攻击
    attack_result = type1_attack(np.expand_dims(data_array, axis=1), start_index, duration)

    print(f"{start_index} - {end_index} : {duration}")

    # 将攻击结果存储到attack_df中
    attack_df.iloc[i, :] = attack_result.squeeze()

    # 更新标签
    label_df.iloc[i, start_index:end_index] = 1

# %%
# 将攻击的结果和标签存储为两个CSV文件
data_save_path = Path('data')
attack_df.to_csv(data_save_path/'attack.csv', index=False)
label_df.to_csv(data_save_path/'label.csv', index=False)
