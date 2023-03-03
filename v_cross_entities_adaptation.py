import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from utils import *

# metric = "precision"
metric = "f1"
# metric = "recall"
# model = "gat"
# model = "gat"
# model = "omni"

# dataset = "MSL"
# dataset = "SMAP"
# end = ".npy"

dataset = "SMD"
end = ".txt"
prefix = "machine-"

# dataset = "SWAT"
# dataset = "WADI"


def get_data(method_path):

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
    dir_path = f"data/{dataset}/train"
    base_path = f"output/{dataset}"
    for file_name in os.listdir(dir_path):
        if file_name.endswith(end):
            group = file_name.split(end)[0]
            group = group.split(prefix)[1] # for smd

    # for WADI
    # groups_term = ["A2"]
    # base_path = f"output/{dataset}"
    # for group in groups_term:
    #     if True:

    # for SWAT
    # groups_term = ["A1_A2", "A4_A5"]
    # groups_term = ["A1_A2"]
    # groups_term = ["A4_A5"]
    # base_path = f"output/{dataset}"
    # for group in groups_term:
    #     if True:

            path = base_path + "/" + group
            file_name2 = path + f"/{method_path}/summary.txt"
            try:
                f1 = get_f1_for_maml(file_name2)
                groups.append(group)
                if metric == "f1":
                    f1s.append(f1)
            except Exception as e:
                print(f"error process in {file_name2}")
                continue
    return groups, f1s


def display_curves(groups, f1s, name):
    mean_f1 = np.mean(f1s)
    print(mean_f1)
    plt.plot(groups, f1s, label=name, marker="^")
    

if __name__ == '__main__':
    open_maml = False
    using_labeled_val = False
    method_path = f"{open_maml}_DC_{using_labeled_val}_Semi"
    groups, f1s = get_data(method_path)
    open_maml = True
    using_labeled_val = False
    method_path = f"{open_maml}_DC_{using_labeled_val}_Semi"
    groups2, f1s2 = get_data(method_path)
    open_maml = False
    using_labeled_val = True
    method_path = f"{open_maml}_DC_{using_labeled_val}_Semi"
    groups3, f1s3 = get_data(method_path)
    open_maml = True
    using_labeled_val = True
    method_path = f"{open_maml}_DC_{using_labeled_val}_Semi"
    groups4, f1s4 = get_data(method_path)
    plt.figure(dpi=300, figsize=(30, 8))
    display_curves(groups, f1s, "FF")
    display_curves(groups2, f1s2, "TF")
    display_curves(groups3, f1s3, "FT")
    display_curves(groups4, f1s4, "TT")
    plt.grid(linestyle="-.")
    plt.legend(loc='best', fontsize=8)
    plt.xlabel("group")
    plt.title(f"{dataset}_reasonable")
    plt.savefig(f"analysis/{dataset}_reasonable_v2.pdf")
