import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import datetime
import warnings

# 读入数据
warnings.filterwarnings("ignore")
resource_pool = "xar03"  # "xar10", "xar03"
# resource_pool = "xar10"  # "xar10", "xar03"
with open(f'source_data/{resource_pool}_relation.json') as f:
    relation = json.load(f)
# inband = pd.read_csv(f'source_data/{resource_pool}/inband.csv', sep=",", header=None, skiprows=1).dropna(axis=0)
# outband = pd.read_csv(f'source_data/{resource_pool}/outband.csv', sep=",", header=None, skiprows=1).dropna(axis=0)
inband = pd.read_csv(f'source_data/{resource_pool}/inband.csv', sep=",").dropna(axis=0)
outband = pd.read_csv(f'source_data/{resource_pool}/outband.csv', sep=",").dropna(axis=0)
limited_diff = 1


def get_limited(lls):
    length = len(lls)
    limited = []
    for i in range(length):
        if i == 0:
            end_date = datetime.datetime.strptime(str(lls[i][1]),'%Y-%m-%d').date()
            limited1 = merged_data[(end_date > merged_data.iloc[:, 0].dt.date) &
                                   (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
                                   (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))] 
            limited1["limited"] = float(lls[i][2])
            limited = limited1
            continue
        elif i == length-1:
            start_date = datetime.datetime.strptime(str(lls[i][0]),'%Y-%m-%d').date()
            limited1 = merged_data[(start_date <= merged_data.iloc[:, 0].dt.date) &
                                   (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
                                   (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
            limited1["limited"] = float(lls[i][2])
            limited = pd.concat([limited, limited1])
        else:
            start_date = datetime.datetime.strptime(str(lls[i][0]),'%Y-%m-%d').date()
            end_date = datetime.datetime.strptime(str(lls[i][1]),'%Y-%m-%d').date()
            limited1 = merged_data[(start_date <= merged_data.iloc[:, 0].dt.date) &
                                   (end_date > merged_data.iloc[:, 0].dt.date) &
                                   (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
                                   (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
            limited1["limited"] = float(lls[i][2])
            limited = pd.concat([limited, limited1])
    return limited


for i in tqdm(relation):
    # 根据机器对应关系配对inband和outband数据
    filtered_inband = inband.loc[inband.iloc[:, 5] == i]
    filtered_outband = outband.loc[outband.iloc[:, 5] == relation[i]]
    if len(filtered_inband) > 0 and len(filtered_outband) > 0:
        # 根据"开始时间"合并inband和outband数据，并按照"开始时间"排序
        # new_inband = filtered_inband.iloc[:, [0, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]]
        new_inband = filtered_inband.iloc[:, [0, 6]]
        # new_outband = filtered_outband.iloc[:, [0, 6, 7]]
        new_outband = filtered_outband.iloc[:, [0, 6]]
        merged_data = pd.merge(new_inband, new_outband, on=list(new_inband)[0])
        merged_data.iloc[:, 0] = pd.to_datetime(merged_data.iloc[:, 0])
        merged_data = merged_data.sort_values(by=list(merged_data)[0], ascending=True)  # sort_values,merge 给列下标是否可以？达没

        # 根据每日功率限制的时间段，标记功率限制标签
        # limited = merged_data[~merged_data.iloc[:, 0].dt.hour.isin(np.arange(0, 13))]
        limited = merged_data[(datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
                              (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
        limited["limited"] = 1
        if limited_diff:
            lls = []
            with open("data/ZX/condition.csv") as f:
                lines = f.readlines()
                for line in lines:
                    lls.append(line.split(","))
            limited = get_limited(lls)

        # unlimited = merged_data[merged_data.iloc[:, 0].dt.hour.isin(np.arange(0, 13))]
        unlimited = merged_data[(merged_data.iloc[:, 0].dt.time <= datetime.time(13)) |
                                (merged_data.iloc[:, 0].dt.time >= datetime.time(23, 30))]
        # merged_data.iloc[:, 0].time <= datetime.time(13) or merged_data.iloc[:, 0].time >= datetime.time(23, 30)
        unlimited["limited"] = 0
        result_data = pd.concat([limited, unlimited])
        result_data = result_data.sort_values(by=list(result_data)[0], ascending=True)
        result_data["aux"] = 0  # 添加辅助列，方便后续数据处理

        # 导出数据
        result_data.iloc[:int(len(result_data) * 0.5), :].to_csv(f"data/ZX/train/{i}.csv", index=False)
        result_data.iloc[int(len(result_data) * 0.5):, :].to_csv(f"data/ZX/test/{i}.csv", index=False)

# todo: 多实体时间序列预测
# todo: metric 制表
# todo: 线性模型效果
# todo: 建立干扰与功率之间的关系 done half

        
        
# limited1 = merged_data[(datetime.date(2023, 2, 7) > merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#                                    (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             limited1["limited"] = 0.1
#             limited2 = merged_data[(datetime.date(2023, 2, 7) <= merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.date(2023, 2, 10) > merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#                                    (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             limited2["limited"] = 0
#             limited3 = merged_data[(datetime.date(2023, 2, 10) <= merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.date(2023, 2, 13) > merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#                                    (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             limited3["limited"] = 0.05
#             # limited4 = merged_data[(datetime.date(2023, 2, 13) <= merged_data.iloc[:, 0].dt.date) &
#             #                        (datetime.date(2023, 2, 17) > merged_data.iloc[:, 0].dt.date) &
#             #                        (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#             #                        (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             # limited4["limited"] = 0.15
#             # limited5 = merged_data[(datetime.date(2023, 2, 17) <= merged_data.iloc[:, 0].dt.date) &
#             #                        (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#             #                        (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             # limited5["limited"] = 0.15
            
#             limited6 = merged_data[(datetime.date(2023, 2, 13) <= merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.date(2023, 3, 10) > merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#                                    (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             limited6["limited"] = 0.15
#             limited7 = merged_data[(datetime.date(2023, 3, 10) <= merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.date(2023, 3, 13) > merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#                                    (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             limited7["limited"] = 0.16
#             limited8 = merged_data[(datetime.date(2023, 3, 13) <= merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.date(2023, 3, 16) > merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#                                    (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             limited8["limited"] = 0.17
            
            
#             limited9 = merged_data[(datetime.date(2023, 3, 16) <= merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.date(2023, 3, 17) > merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#                                    (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             limited9["limited"] = 0.19
#             limited10 = merged_data[(datetime.date(2023, 3, 17) <= merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.date(2023, 3, 20) > merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#                                    (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             limited10["limited"] = 0.20
#             limited11 = merged_data[(datetime.date(2023, 3, 20) <= merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.date(2023, 3, 22) > merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#                                    (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             limited11["limited"] = 0.19
#             limited12 = merged_data[(datetime.date(2023, 3, 22) <= merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.date(2023, 3, 25) > merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#                                    (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             limited12["limited"] = 0.18
#             limited13 = merged_data[(datetime.date(2023, 3, 25) <= merged_data.iloc[:, 0].dt.date) &
#                                    (datetime.time(13) <= merged_data.iloc[:, 0].dt.time) &
#                                    (merged_data.iloc[:, 0].dt.time <= datetime.time(23, 30))]
#             limited13["limited"] = 0.15
#             limited = pd.concat([limited1, limited2, limited3, limited4, limited5])