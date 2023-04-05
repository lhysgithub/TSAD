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
select_day = 4  # 16
# anomaly_days = [9,10]
select_days = [10,11] # [2,3,4] [7,8,9] [10,11,12] [13,14,15]

# 如果需要遍历所有实体，从此处开启循环
train_data = pd.read_csv(f'data/ZX/train/{server_id}.csv', sep=",").dropna(axis=0)
test_data = pd.read_csv(f'data/ZX/test/{server_id}.csv', sep=",").dropna(axis=0)
train_data.iloc[:, 0] = pd.to_datetime(train_data.iloc[:, 0]) + datetime.timedelta(hours=8)
test_data.iloc[:, 0] = pd.to_datetime(test_data.iloc[:, 0]) + datetime.timedelta(hours=8)
train_data.set_index(list(train_data)[0], inplace=True)
test_data.set_index(list(test_data)[0], inplace=True)
data = pd.concat([train_data, test_data])
if select_day:
    data = data[(data.index.day == select_days[0]) | (data.index.day == select_days[1])]
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

t = data.index.time
mask3 = ((t <= datetime.time(pink_time,30)) & (t >= datetime.time(pink_time-1,30)))
mask4 = ((t <= datetime.time(pink_time_2+1)) & (t >= datetime.time(pink_time_2)))
index_list_3, lens_list_3 = find_positive_segment(mask3)
index_list_4, lens_list_4 = find_positive_segment(mask4)
plt.figure(figsize=(8 * shape, 6 * shape))
labels = ["CPU Load (%)", "Storage Read IOPS (IO/s)", "Storage Write IOPS (IO/s)", "Network In Usage (%)",
          "Network Out Usage (%)", "Hypervisor CPU Load (%)", "Hypervisor Memory Usage (%)", "Storage Usage (%)",
          "Storage Read Latency (ms)", "Storage Write Latency (ms)", "Power (W)", "CPU Temperature (°C)",
          "Power Limited", "Aux"]
target_index = [0, 10, 9]
time_labels = range(13)
for i in range(len(target_index)):
    target_dim = target_index[i]
    plt.subplot(3, 1, i+1)
    column = data.iloc[:, target_dim]
    plt.plot(range(len(column)), column, label=f"{labels[target_dim]}")
    plt.xticks(range(0, len(column), 12*3)[:16], range(0, 24*2, 3))
    y_max = np.max(column)
    y_min = np.min(column)
    plt.ylabel(labels[target_dim].split("(")[0])
    if i ==2:
        plt.xlabel("Time (Hour)")
    # for k in range(len(index_list_3)):
    for k in range(0, 1):
        start = index_list_3[k]
        end = index_list_3[k] + lens_list_3[k]
        plt.fill_between([start, end], y_min, y_max, facecolor='red', alpha=0.5)
    for k in range(1, len(index_list_4)):
        start = index_list_4[k]
        end = index_list_4[k] + lens_list_4[k]
        plt.fill_between([start, end], y_min, y_max, facecolor='red', alpha=0.5)
    # for k in range(len(index_list_4)):
    #     start = index_list_4[k]
    #     end = index_list_4[k] + lens_list_4[k]
    #     plt.fill_between([start, end], y_min, y_max, facecolor='green', alpha=0.5)
    plt.legend()
    # plt.title(f"zx_{server_id}_{select_days}")
plt.savefig(f"analysis/zx_{server_id}_{select_days}_anomaly_example.pdf")
