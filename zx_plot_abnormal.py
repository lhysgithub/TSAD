from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import datetime

# 忽视在模型拟合中遇到的错误
warnings.filterwarnings("ignore")
shape = 2
target_dim = 0  # 10: 功率，0: CPU使用
iterm = "Power" if target_dim == 10 else "CPU Usage"
# 导入数据
resource_pool = "xar03"
# server_id = "computer-b0504-05"
server_id = "computer-b0503-01"
select_day = 12  # 16

# 如果需要遍历所有实体，从此处开启循环
train_data = pd.read_csv(f'data/ZX/train/{server_id}.csv', sep=",").dropna(axis=0)
test_data = pd.read_csv(f'data/ZX/test/{server_id}.csv', sep=",").dropna(axis=0)
train_data.iloc[:, 0] = pd.to_datetime(train_data.iloc[:, 0]) + datetime.timedelta(hours=8)
test_data.iloc[:, 0] = pd.to_datetime(test_data.iloc[:, 0]) + datetime.timedelta(hours=8)
train_data.set_index(list(train_data)[0], inplace=True)
test_data.set_index(list(test_data)[0], inplace=True)
# data = train_data.iloc[:, target_dim]
# data = test_data.iloc[:, target_dim]
data = pd.concat([train_data, test_data])
if select_day:
    data = data[data.index.day == select_day]
pink_time = 21
pink_time_2 = 7


def find_positive_segment(data):
    index_list = []
    lens_list = []
    find = 0
    count = 0
    for i in range(len(data)):
        if int(data[i]) == 1:
            if find == 0:
                index_list.append(i)
            find = 1
            count += 1
        elif find == 1:
            find = 0
            lens_list.append(count)
            count = 0
    return index_list, lens_list


# mask2 = ((data.index.hour <= pink_time) & (data.index.hour >= pink_time - 1)) | ((data.index.hour <= pink_time_2) & (data.index.hour >= pink_time_2 - 1))
t = data.index.time
mask2 = ((t <= datetime.time(pink_time,30)) & (t >= datetime.time(pink_time-1,30))) | ((t <= datetime.time(pink_time_2+1)) & (t >= datetime.time(pink_time_2)))
index_list, lens_list = find_positive_segment(mask2)
plt.figure(figsize=(8 * shape, 6 * shape))
labels = ["CPU Load (%)", "Storage Read IOPS (IO/s)", "Storage Write IOPS (IO/s)", "Network In Usage (%)",
          "Network Out Usage (%)", "Hypervisor CPU Load (%)", "Hypervisor Memory Usage (%)", "Storage Usage (%)",
          "Storage Read Latency (ms)", "Storage Write Latency (ms)", "Power (W)", "CPU Temperature (°C)",
          "Power Limited", "Aux"]
# for j in range(len(list(data))):
#     plt.cla()
#     column = data.iloc[:, j]
#     plt.plot(range(len(column)), column, label=f"{labels[j]}")
#     y_max = np.max(column)
#     y_min = np.min(column)
#     for i in range(len(index_list)):
#         start = index_list[i]
#         end = index_list[i] + lens_list[i]
#         plt.fill_between([start, end], y_min, y_max, facecolor='red', alpha=0.5)
#     plt.legend()
#     plt.savefig(f"analysis/zx_{server_id}_{j}.pdf")

target_index = [0, 10, 11]
time_labels = range(13)
for i in range(len(target_index)):
    target_dim = target_index[i]
    plt.subplot(3, 1, i+1)
    column = data.iloc[:, target_dim]
    plt.plot(range(len(column)), column, label=f"{labels[target_dim]}")
    # plt.xticks([int(i/12) for i in range(1, len(column)+1, 12*4)])
    plt.xticks(range(0, len(column), 12 * 3), range(0, 24, 3))
    y_max = np.max(column)
    y_min = np.min(column)
    plt.ylabel(labels[target_dim].split("(")[0])
    if i ==2:
        plt.xlabel("Time (Hour)")
    for k in range(len(index_list)):
        start = index_list[k]
        end = index_list[k] + lens_list[k]
        plt.fill_between([start, end], y_min, y_max, facecolor='red', alpha=0.5)
    plt.legend()
plt.savefig(f"analysis/zx_{server_id}_anomaly_example.pdf")
