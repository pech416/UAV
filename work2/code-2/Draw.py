import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def data_show(total_t, rewards, alphas, losses):
    mean_value = [total_t, rewards, alphas, losses]
    x_label = 'Episode'
    y_labels = ['Delay(s)', 'Reward', 'Alpha', 'Loss']
    fig, axs = plt.subplots(nrows=2, ncols=2, dpi=120)
    for i in range(4):
        axs[int(i / 2), i % 2].plot(range(len(mean_value[i])), mean_value[i], color="r")
        axs[int(i / 2), i % 2].set_xlabel(x_label)
        axs[int(i / 2), i % 2].set_ylabel(y_labels[i])
        axs[int(i / 2), i % 2].grid(linewidth=0.5, zorder=-10)
    plt.tight_layout()
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.show()
    return

file = "UAV.xlsx"

data1 = pd.read_excel(file, 'Sheet1', usecols=['times'])
data2 = pd.read_excel(file, 'Sheet1', usecols=['reward'])
data3 = pd.read_excel(file, 'Sheet1', usecols=['alpha'])
data4 = pd.read_excel(file, 'Sheet1', usecols=['losses'])

episode = 3000
delay, reward, alpha, loss = [], [], [], []
for i in range(episode):
    delay.append(data1.values[i][0])
for i in range(episode):
    reward.append(data2.values[i][0])
for i in range(episode):
    alpha.append(data3.values[i][0])
for i in range(episode):
    loss.append(data4.values[i][0])

delay_, reward_, alpha_, loss_ = [], [], [], []
nums = 10
for i in range(episode - nums):
    delay_.append(np.mean(delay[i:nums+i]))
for i in range(episode - nums):
    reward_.append(np.mean(reward[i:nums+i]))
for i in range(episode - nums):
    alpha_.append(np.mean(alpha[i:nums+i]))
for i in range(episode - nums):
    loss_.append(np.mean(loss[i:nums+i]))

data_show(delay_, reward_, alpha_, loss_)