import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from utils import *

dataset = "SWAT"
# dataset = "MSL"
# dataset = "SMAP"
# dataset = "SMD"

# method_type = "maml_un"
method_type = "maml_val"
# method_type = "baseline_gat"
# method_type = "baseline_omni"

# target_entry = "overall"
target_entry = "A4_A5"
groups_term = [target_entry]

model_type = "norm"
# model_type = "wo_norm"

# metric = "precision"
# metric = "f1"
metric = "recall"

model = "gat"
# model = "omni"

def get_data():

    f1s = []
    groups = []

    # for SMAP MSL
    # end = ".npy"
    # dir_path = f"data/SMAP/train"
    # base_path = f"output/{dataset}"
    # csv_path = f"data/SMAP/{dataset.lower()}_train_md.csv"
    # meta_data = pd.read_csv(csv_path, sep=",", index_col="chan_id")
    # for group in meta_data.index:
    #     if "-" in group:

    # for SMD
    # end = ".txt"
    # prefix = "machine-"
    # dir_path = f"data/{dataset}/train"
    # base_path = f"output/{dataset}"
    # for file_name in os.listdir(dir_path):
    #     if file_name.endswith(end):
    #         group = file_name.split(end)[0]
    #         group = group.split(prefix)[1] # for smd

    # for WADI
    # dataset = "WADI"
    # groups_term = ["A2"]
    # base_path = f"output/{dataset}"
    # for group in groups_term:
    #     if True:

    # for SWAT
    # groups_term = ["A1_A2", "A4_A5"]
    # groups_term = ["A1_A2"]
    base_path = f"output/{dataset}"
    for group in groups_term:
        for i in range(5):
            target_dir = f"{method_type}_{model_type}_{i}"

            path = base_path + "/" + group
            file_name2 = path + f"/{target_dir}/summary.txt"
            try:
                f1 = get_key_for_maml(file_name2, metric)
                # f1 = get_f1_for_omni_broken(file_name2)
                if f1 < 0.1:
                    continue
                # f1 = get_f1(file_name2)
                # recall = get_recall(file_name2)
                # precision = get_precision(file_name2)
                groups.append(target_dir)
                # if metric == "f1":
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
    max_f1 = np.max(f1s)
    print(mean_f1)
    plt.figure(dpi=300, figsize=(30, 8))
    plt.plot(groups, f1s, c="r", label=f'f1', marker="^")
    # plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.grid(linestyle="-.")
    plt.legend(loc='best', fontsize=8)
    plt.xlabel("group")
    name = f"{dataset}_{model}_{method_type}_{model_type}_{target_entry}_{metric}_{mean_f1:.4f}_{max_f1:.4f}"
    plt.title(f"{name}")
    plt.savefig(f"analysis/{name}.pdf")
