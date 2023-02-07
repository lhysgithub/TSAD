from utils import *
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import product
from tqdm import tqdm_notebook, tqdm
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from datetime import timedelta
import datetime

# 忽视在模型拟合中遇到的错误
warnings.filterwarnings("ignore")
shape = 2
target_dim = 0  # 10: 功率，0: CPU使用
iterm = "Power" if target_dim == 10 else "CPU Usage"
# 导入数据
resource_pool = "xar03"
server_id = "computer-b0504-05"
# server_id = "computer-b0503-01"
select_day = 12 # 16

# 如果需要遍历所有实体，从此处开启循环
train_data = pd.read_csv(f'data/ZX/train/{server_id}.csv', sep=",").dropna(axis=0)
test_data = pd.read_csv(f'data/ZX/test/{server_id}.csv', sep=",").dropna(axis=0)
train_data.iloc[:, 0] = pd.to_datetime(train_data.iloc[:, 0]) + datetime.timedelta(hours=8)
test_data.iloc[:, 0] = pd.to_datetime(test_data.iloc[:, 0]) + datetime.timedelta(hours=8)
train_data.set_index(list(train_data)[0], inplace=True)
test_data.set_index(list(test_data)[0], inplace=True)
# data = train_data.iloc[:, target_dim]
# data = test_data.iloc[:, target_dim]
data = pd.concat([train_data, test_data]).iloc[:, target_dim]
if select_day:
    data = data[data.index.day==select_day]
pink_time = 21


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


mask2 = ((data.index.hour <= pink_time) & (data.index.hour >= pink_time - 1))
index_list, lens_list = find_positive_segment(mask2)
plt.figure(figsize=(8 * shape, 6 * shape))
plt.plot(data.index, data.values, label=iterm, color='blue')
y_max = np.max(data)
y_min = np.min(data)
for i in range(len(index_list)):
    plt.fill_between(
        [data.index[index_list[i] + int(lens_list[i] / 2)], data.index[index_list[i] + int(lens_list[i] / 2) + 1]],
        y_min, y_max, facecolor='red',
        alpha=0.5)
    plt.fill_between([data.index[index_list[i]], data.index[index_list[i] + lens_list[i]]], y_min, y_max,
                     facecolor='pink', alpha=0.5)
plt.legend()
plt.savefig(f'output/ZX/analysis/{iterm}_change_while_power_limited_{pink_time}_{server_id}_{select_day}.pdf')
