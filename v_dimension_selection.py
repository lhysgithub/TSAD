import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from utils import *


metric = "f1"
# metric = "ree"
# metric = "precision"
# metric = "recall"
# aggregate = "mean"
aggregate = "max"

dataset = "SMD"
entity = "1-8"
compose_number = 38
feature_number = 38
gap = 2
base_path = "search"

# dataset = "SMAP"
# end = ".npy"
# dataset = "SMD"
# end = ".txt"
# prefix = "machine-"

# dataset = "SMAP"
# entity = "A-7"
# compose_number = 25
# feature_number = 25
# gap = 2
# base_path = "20221110_smap"

# dataset = "WT"
# entity = "WT23"
# compose_number = 10
# feature_number = 10
# gap = 1
# base_path = "20221110_wt"


def get_data_all(metric,base_path):
    f1s = []
    composes = []
    for k in range(1,compose_number+1):
        temp = []
        temp_com = []
        for currentId in range(1,pow(2,feature_number)):
            current_id_bin = "{:0>10b}".format(currentId)
            best_path = base_path + "_" + current_id_bin
            file_name = f"./output/{dataset}/{entity}/{best_path}/summary_file.txt"
            try:  # 对于max_selection，很多组合都没遍历到，所以要尝试性的访问文件
                f1 = get_f1(file_name)
            except Exception as e:
                continue

            # 计算维度组合数量
            selected = []
            for i in range(10):
                if current_id_bin[i] == '1':
                    selected.append(i)

            # 维度组合数符合要求，则加入集合
            if len(selected) == k:
                temp.append(f1)
                temp_str = [str(w) for w in selected]
                temp_com.append("_".join(temp_str))
        f1s.append(temp)
    f1s_agg = []
    for l in f1s:
        f1s_agg.append(np.max(l))
    return f1s_agg, f1s, composes


def get_data(metric,base_path):
    f1s = []
    composes = []
    selected = []
    best_path = base_path
    # best_list = [0,3,10,12,13,8,15,9,27,26,37,21,11,14,7,5,16,31,4,34,20,17,18,35,36,23,6,28,2,29,30,24,1,32,19,25,22,33]
    best_list = [15,12,13,2,14,9,23,4,8,26,0,37,7,36,33,1,11,17,34,5,6,10,28,24,18,30,31,25,16,19,32,27,20,3,21,35,22,29]
    for i in range(compose_number):
        temp2_f1 = []
        for j in range(feature_number):
            if j in selected:
                continue
            temp_select = selected + [j]
            current_id_bin = list2bin(temp_select,feature_number)
            best_path = base_path + "_" + current_id_bin
            file_name = f"./output/{dataset}/{entity}/{best_path}/summary.txt"
            f1 = get_f1(file_name)
            precision = get_precision(file_name)
            recall = get_recall(file_name)
            if metric == "f1":
                temp2_f1.append(f1)
            elif metric == "precision":
                temp2_f1.append(precision)
            elif metric == "recall":
                temp2_f1.append(recall)
        mean_f1 = np.array(temp2_f1).mean()

        best_f1 = 0
        best_id = -1
        temp_f1 = []
        temp_com = []
        for j in range(feature_number):
            if j in selected:
                temp_f1.append(mean_f1)
                selected_str = [str(k) for k in selected]
                current_composition = "_".join(selected_str)
                temp_com.append(current_composition)
                continue
            temp_select = selected + [j]
            current_id_bin = list2bin(temp_select, feature_number)
            best_path = base_path + "_" + current_id_bin
            file_name = f"./output/{dataset}/{entity}/{best_path}/summary.txt"
            f1 = get_f1(file_name)
            precision = get_precision(file_name)
            recall = get_recall(file_name)
            selected_str = [str(k) for k in selected]
            if len(selected_str) == 0:
                current_composition = "_".join(selected_str) + f"{j}"
            else:
                current_composition = "_".join(selected_str) + f"_{j}"
            temp_com.append(current_composition)
            if metric == "f1":
                temp_f1.append(f1)
            elif metric == "precision":
                temp_f1.append(precision)
            elif metric == "recall":
                temp_f1.append(recall)
            if best_f1 < f1:
                best_f1 = f1
                best_id = j
        f1s.append(temp_f1)
        composes.append(temp_com)
        if best_id == -1:
            best_id = best_list[i]
        best_path = best_path + f"_{best_id}"
        selected = selected + [best_id]
    f1s_np = np.array(f1s)
    coms_np = composes
    f1s_np_max = np.max(f1s_np, axis=1)  # mean or max?
    f1s_np_mean = np.mean(f1s_np, axis=1)  # mean or max?
    if aggregate == "mean":
        f1s_np_max = f1s_np_mean
    return f1s_np_max,f1s_np,coms_np


def render_heatmap(data_np,composition,type):
    data_pd = pd.DataFrame()
    for i in range(len(data_np)):
        data_pd.insert(len(data_pd.columns), i, data_np[i])

    plt.figure()
    # sns.heatmap(data_pd)
    center = np.mean(data_np)#0.5
    labels = []
    temp_labels = [i + 1 for i in range(feature_number)]
    for i in temp_labels:
        if i % gap == 0:
            labels.append(i)
        else:
            labels.append("")
    p = sns.heatmap(data_pd,  cmap="RdBu_r", xticklabels=labels, yticklabels=labels,
                    cbar_kws={"label": f"{type}"}, vmin=0, vmax=1, center=center,
                    square=False)#yticklabels=ylabes,
    p.set_ylabel("index")
    p.set_xlabel("composition")
    plt.title(f"{dataset}_{entity}_{type}_heatmap")
    plt.savefig(f"analysis/{dataset}_{entity}_{compose_number}_{type}_heatmap.pdf")


def render_csv(data_np, composition, type):
    data_com_pd = pd.DataFrame()
    for i in range(len(data_np)):
        data_com_pd.insert(len(data_com_pd.columns), f"{i}_coms", composition[i])
        data_com_pd.insert(len(data_com_pd.columns), f"{i}_f1", data_np[i])
    data_com_pd.to_csv(f"analysis/{dataset}_{entity}_{compose_number}_{type}_composition.csv", float_format='%.2f')


# max_selection 和 complete_composition 需要使用不同的文件名
if __name__ == '__main__':
    f1s_agg, f1s,composes = get_data("f1", "search_norm")
    plt.figure()
    plt.plot(np.arange(len(f1s_agg)) + 1, f1s_agg, c="b", label=f'f1_all_composition', marker="v")
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.grid(linestyle="-.")
    plt.legend(loc='best', fontsize=8)
    plt.title(f"{dataset}_{entity}_{aggregate}_norm_all_tendency")
    plt.savefig(f"analysis/{dataset}_{entity}_{compose_number}_{aggregate}_norm_all_tendency.pdf")

    render_heatmap(f1s, composes, "f1")
    render_csv(f1s, composes, "f1")
