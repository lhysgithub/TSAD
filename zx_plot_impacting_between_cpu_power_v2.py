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
pink_time = 21
pink_time_2 = 7
# 导入数据
resource_pool = "xar03"
# server_id = "computer-b0504-05"

# select_day = 4  # 16
# select_days = [7,8,9] # [2,3,4] [7,8,9] [10,11,12] [13,14,15]
five_select_days = [10,11,12]
ten_select_days = [2,3,4]
zero_select_days = [7,8,9]
fifteen_select_days = [13,14,15]
# anomaly_days = [9,10]
select_days = fifteen_select_days # zero_select_days five_select_days + ten_select_days + fifteen_select_days [2,3,4,10,11,12,13,14,15]

def comput_diff(after_i,before_i,data,dim,select_days):
    sum = 0
    count = 0
    for i in select_days:
        # data_ = data[data.index.day == i]
        after_data = data.iloc[:, dim][after_i]
        before_data = data.iloc[:, dim][before_i]
        local_mean = after_data[after_data.index.day == i].mean() - before_data[before_data.index.day == i].mean()
        sum = sum + local_mean
        if local_mean > 0 :
            count = count + 1
    # print(count)
    return sum / len(select_days), count*1.0/len(select_days)

def comput_max_diff(after_i,before_i,data,dim,select_days):
    sum = 0
    count = 0
    for i in select_days:
       # data_ = data[data.index.day == i]
       after_data = data.iloc[:, dim][after_i]
       before_data = data.iloc[:, dim][before_i]
       local_max = after_data[after_data.index.day == i].max() - before_data[before_data.index.day == i].max()
       sum = sum+local_max
       if local_max > 0:
           count = count + 1
    # print(count)
    return sum/len(select_days), count*1.0/len(select_days)

with open("source_data/xar03_relation.json") as f:
    xar03 = json.load(f)

lds, uds, lups, uups = [],[],[],[]
lmds, umds, lmups, umups = [],[],[],[]

for server_id in xar03.keys():
    try:
        # server_id = "computer-b0503-01"
        # 如果需要遍历所有实体，从此处开启循环
        train_data = pd.read_csv(f'data/ZX/train/{server_id}.csv', sep=",").dropna(axis=0)
        test_data = pd.read_csv(f'data/ZX/test/{server_id}.csv', sep=",").dropna(axis=0)
        train_data.iloc[:, 0] = pd.to_datetime(train_data.iloc[:, 0]) + datetime.timedelta(hours=8)
        test_data.iloc[:, 0] = pd.to_datetime(test_data.iloc[:, 0]) + datetime.timedelta(hours=8)
        train_data.set_index(list(train_data)[0], inplace=True)
        test_data.set_index(list(test_data)[0], inplace=True)
        data = pd.concat([train_data, test_data])

        t = data.index.time
        after_limited_i = ((t <= datetime.time(pink_time,30)) & (t >= datetime.time(pink_time)))
        before_limited_i = ((t <= datetime.time(pink_time)) & (t >= datetime.time(pink_time-1,30)))
        after_unlimited_i = ((t <= datetime.time(pink_time_2+1)) & (t >= datetime.time(pink_time_2,30)))
        before_unlimited_i = ((t <= datetime.time(pink_time_2,30)) & (t >= datetime.time(pink_time_2)))

        limited_diff,l_upper_p = comput_diff(after_limited_i,before_limited_i,data,0,select_days)
        unlimited_diff,u_upper_p = comput_diff(after_unlimited_i,before_unlimited_i,data,0,select_days)
        # print(f"limited_diff: {limited_diff} upp: {l_upper_p}")
        # print(f"unlimited_diff: {unlimited_diff} downp: {1-u_upper_p}")
        lds.append(limited_diff)
        uds.append(unlimited_diff)
        lups.append(l_upper_p)
        uups.append(u_upper_p)

        limited_max_diff,l_max_upper_p = comput_max_diff(after_limited_i,before_limited_i,data,0,select_days)
        unlimited_max_diff,u_max_upper_p = comput_max_diff(after_unlimited_i,before_unlimited_i,data,0,select_days)
        # print(f"limited_max_diff: {limited_max_diff} upp: {l_max_upper_p}")
        # print(f"unlimited_max_diff: {unlimited_max_diff} downp: {1-u_max_upper_p}")
        lmds.append(limited_max_diff)
        umds.append(unlimited_max_diff)
        lmups.append(l_max_upper_p)
        umups.append(u_max_upper_p)
    except Exception as e:
        print(f"server_id: {server_id}")
        continue

print(f"limited_diff: {np.array(lds).mean()} upp: {np.array(lups).mean()}")
print(f"unlimited_diff: {np.array(uds).mean()} downp: {1-np.array(uups).mean()}")
print(f"limited_max_diff: {np.array(lmds).mean()} upp: {np.array(lmups).mean()}")
print(f"unlimited_max_diff: {np.array(umds).mean()} downp: {1-np.array(umups).mean()}")