import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from utils import *

# method_path = "baseline_gat"
# method_path = "baseline_gat_norm_2"
method_path = "maml_un"
# method_path = "maml_val"
target_entry = "overall"

# metric = "precision"
metric = "f1"
# metric = "recall"
# model = "gat"
model = "gat"
# model = "omni"

# dataset = "MSL"
# dataset = "SMAP"
# end = ".npy"

# dataset = "SMD"
# end = ".txt"
# prefix = "machine-"
#
dataset = "SWAT"
# dataset = "WADI"


def get_data():

    f1s = []
    groups = []

    # # for SMAP MSL
    # dir_path = f"data/SMAP/train"
    # base_path = f"output/{dataset}"
    # csv_path = f"data/SMAP/{dataset.lower()}_train_md.csv"
    # meta_data = pd.read_csv(csv_path, sep=",", index_col="chan_id")
    # for group in meta_data.index:
    #     if "-" in group:

    # for SMD
    # dir_path = f"data/{dataset}/train"
    # base_path = f"output/{dataset}"
    # for file_name in os.listdir(dir_path):
    #     if file_name.endswith(end):
    #         group = file_name.split(end)[0]
    #         group = group.split(prefix)[1] # for smd

    # for WADI
    # groups_term = ["A2"]
    # base_path = f"output/{dataset}"
    # for group in groups_term:
    #     if True:

    # for SWAT
    # groups_term = ["A1_A2", "A4_A5"]
    # groups_term = ["A1_A2"]
    groups_term = ["A4_A5"]
    base_path = f"output/{dataset}"
    for group in groups_term:
        if True:

            path = base_path + "/" + group
            file_name2 = path + f"/{method_path}/summary.txt"
            try:
                # f1 = get_f1_for_maml(file_name2)
                f1 = get_f1_for_omni_broken(file_name2)
                # f1 = get_f1(file_name2)
                # recall = get_recall(file_name2)
                # precision = get_precision(file_name2)
                groups.append(group)
                if metric == "f1":
                    f1s.append(f1)
                # elif metric == "recall":
                #     f1s.append(recall)
                # elif metric == "precision":
                #     f1s.append(precision)
            except Exception as e:
                print(f"error process in {file_name2}")
                continue
    return groups, f1s


if __name__ == '__main__':
    groups, f1s = get_data()
    mean_f1 = np.mean(f1s)
    print(mean_f1)
    plt.figure(dpi=300, figsize=(30, 8))
    plt.plot(groups, f1s, c="r", label=f'f1', marker="^")
    # plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.grid(linestyle="-.")
    plt.legend(loc='best', fontsize=8)
    plt.xlabel("group")
    plt.title(f"{dataset}_{model}_{method_path}_{target_entry}_{metric}")
    plt.savefig(f"analysis/{dataset}_{model}_{method_path}_{target_entry}_{metric}_{mean_f1}.pdf")

# precision = TP / (TP + FP + 0.00001)
#     recall = TP / (TP + FN + 0.00001)
#     f1 = 2 * precision * recall / (precision + recall + 0.00001)