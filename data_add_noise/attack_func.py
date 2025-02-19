# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def type1_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
    alpha = np.random.uniform(*alpha_range)
    end_point = start_point + duration
    data[:, :, start_point:end_point] *= alpha
    return data

def type2_attack(data, start_point, duration):
    end_point = start_point + duration
    sigma_range = (0, np.max(data))
    sigma = np.random.uniform(*sigma_range)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, start_point:end_point] = np.clip(data[i, j, start_point:end_point], None, sigma)
    return data

def type3_attack(data, start_point, duration):
    end_point = start_point + duration
    gamma_range = (0, np.max(data))
    gamma = np.random.uniform(*gamma_range)
    data[:, :, start_point:end_point] -= gamma
    data = np.maximum(data, 0)  # Ensure no negative values
    return data

def type4_attack(data, start_point, duration):
    end_point = start_point + duration
    data[:, :, start_point:end_point] = 0
    return data

def type5_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
    end_point = start_point + duration
    for t in range(start_point, end_point):
        alpha_t = np.random.uniform(*alpha_range)
        data[:, :, t] *= alpha_t
    return data

def type6_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
    end_point = start_point + duration
    for t in range(start_point, end_point):
        alpha_t = np.random.uniform(*alpha_range)
        daily_average = np.mean(data[:, :, :], axis=2, keepdims=True)
        data[:, :, t] = alpha_t * daily_average[:, :, 0]
    return data

# %%
# Plotting function
def plot_attack(original, attacked, attack_type):
    plt.figure(figsize=(10, 6))
    plt.plot(original.flatten(), label='Original')
    plt.plot(attacked.flatten(), label=f'After {attack_type} Attack')
    plt.legend()
    plt.title(f'{attack_type} Attack')
    plt.xlabel('Time Interval')
    plt.ylabel('Electricity Consumption')
    plt.savefig(f'{attack_type}_attack.png')
    plt.show()

if __name__ == '__main__':
    # Generate example data (1 batch, 1 channel, 48 data points)
    data = np.random.rand(2, 1, 90) * 100
    original_data = data.copy()

    start_point = 10
    duration = 50

    # Apply attacks
    data_type1 = type1_attack(data.copy(), start_point, duration)
    data_type2 = type2_attack(data.copy(), start_point, duration)
    data_type3 = type3_attack(data.copy(), start_point, duration)
    data_type4 = type4_attack(data.copy(), start_point, duration)
    data_type5 = type5_attack(data.copy(), start_point, duration)
    data_type6 = type6_attack(data.copy(), start_point, duration)

    # Plot and save figures
    plot_attack(original_data, data_type1, 'Type 1')
    plot_attack(original_data, data_type2, 'Type 2')
    plot_attack(original_data, data_type3, 'Type 3')
    plot_attack(original_data, data_type4, 'Type 4')
    plot_attack(original_data, data_type5, 'Type 5')
    plot_attack(original_data, data_type6, 'Type 6')