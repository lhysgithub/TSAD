import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from utils import *

# method_path = "baseline"
# method_path = "baseline_omni"
# method_path = "baseline_gat_norm"
# method_path = "maml_un_1220_uni_train_norm"
# method_path = "maml_un_1222_uni_train_stgat_norm"
# method_path = "maml_val_1223_uni_train_stgat_norm"
method_path = "maml_val_1223_multi_train_stgat_norm"
# method_path = "maml_un_1223_multi_train_stgat_norm"
# method_path = "maml_un_1222_uni_train_stgat_norm"
# method_path = "maml_val_norm"
# method_path = "baseline_stgat_0"
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

dataset = "SWAT"
# dataset = "WADI"


def get_data():
    f1s, groups, tps, tns, fps, fns, rs, ps = [], [], [], [], [], [], [], []
    # for SMD
    # dir_path = f"data/{dataset}/train"
    # base_path = f"output/{dataset}"
    # for file_name in os.listdir(dir_path):
    #     if file_name.endswith(end):
    #         group = file_name.split(end)[0]
    #         group = group.split(prefix)[1]  # for smd

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

    # for SMAP MSL
    # base_path = f"output/SMAP"
    # base_path = f"output/{dataset}"
    # csv_path = f"data/SMAP/{dataset.lower()}_train_md.csv"
    # meta_data = pd.read_csv(csv_path, sep=",", index_col="chan_id")
    # for group in meta_data.index:
    #     if "-" in group:

            path = base_path + "/" + group
            temps = []
            temp_ps = []
            temp_rs = []
            for i in range(0,3):
                # if True:
                file_name2 = path + f"/{method_path}_{i}/summary.txt"
                try:
                    f1 = get_key_for_maml(file_name2, "f1")
                    # tp = get_key_for_maml(file_name2, "TP")
                    # tn = get_key_for_maml(file_name2, "TN")
                    # fp = get_key_for_maml(file_name2, "FP")
                    # fn = get_key_for_maml(file_name2, "FN")
                    p = get_key_for_maml(file_name2, "precision")
                    r = get_key_for_maml(file_name2, "recall")
                    temp_rs.append(r)
                    temp_ps.append(p)
                    # f1s.append(get_key_for_maml(file_name2, "f1"))
                    # tps.append(get_key_for_maml(file_name2, "TP"))
                    # tns.append(get_key_for_maml(file_name2, "TN"))
                    # fps.append(get_key_for_maml(file_name2, "FP"))
                    # fns.append(get_key_for_maml(file_name2, "FN"))
                    # ps.append(get_key_for_maml(file_name2, "precision"))
                    # rs.append(get_key_for_maml(file_name2, "recall"))
                    # f1 = get_key_from_bf_result(file_name2, "f1")
                    # f1s.append(get_key_from_bf_result(file_name2, "f1"))
                    # tps.append(get_key_from_bf_result(file_name2, "TP"))
                    # tns.append(get_key_from_bf_result(file_name2, "TN"))
                    # fps.append(get_key_from_bf_result(file_name2, "FP"))
                    # fns.append(get_key_from_bf_result(file_name2, "FN"))
                    # ps.append(get_key_from_bf_result(file_name2, "precision"))
                    # rs.append(get_key_from_bf_result(file_name2, "recall"))
                    # f1 = get_key_for_omni(file_name2, "best-f1")
                    # f1s.append(get_key_for_omni(file_name2, "best-f1"))
                    # tps.append(get_key_for_omni(file_name2, "TP"))
                    # tns.append(get_key_for_omni(file_name2, "TN"))
                    # fps.append(get_key_for_omni(file_name2, "FP"))
                    # fns.append(get_key_for_omni(file_name2, "FN"))
                    # ps.append(get_key_for_omni(file_name2, "precision"))
                    # rs.append(get_key_for_omni(file_name2, "recall"))
                    # f1 = get_key_from_omni_broken(file_name2, "best-f1")
                    # f1s.append(get_key_from_omni_broken(file_name2, "best-f1"))
                    # # tps.append(get_key_from_omni_broken(file_name2, "TP"))
                    # # tns.append(get_key_from_omni_broken(file_name2, "TN"))
                    # # fps.append(get_key_from_omni_broken(file_name2, "FP"))
                    # # fns.append(get_key_from_omni_broken(file_name2, "FN"))
                    # ps.append(get_key_from_omni_broken(file_name2, "precision"))
                    # rs.append(get_key_from_omni_broken(file_name2, "recall"))
                    temps.append(f1)

                except Exception as e:
                    print(f"error process in {file_name2}")
                    continue
            if len(temps) > 0:
                groups.append(group)
            f1s.append(np.array(temps).max())
            ps.append(np.max(temp_ps))
            rs.append(np.max(temp_rs))
    return groups, f1s, tps, tns, fps, fns, ps, rs


if __name__ == '__main__':
    groups, f1s, tps, tns, fps, fns, ps, rs = get_data()
    mean_f1 = np.mean(f1s)
    # print(mean_f1)
    tps_sum = np.array(tps).sum()
    tns_sum = np.array(tns).sum()
    fps_sum = np.array(fps).sum()
    fns_sum = np.array(fns).sum()
    p = tps_sum / (tps_sum + fps_sum + 0.00001)
    r = tps_sum / (tps_sum + fns_sum + 0.00001)
    f1 = 2 * p * r / (p + r + 0.00001)
    print(f"p:{p} r:{r} f1:{f1}")
    print(f"mp:{np.mean(ps)} mr:{np.mean(rs)} mf1:{mean_f1}")
    plt.figure(dpi=300, figsize=(30, 8))
    plt.plot(groups, f1s, c="r", label=f'f1', marker="^")
    # plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.grid(linestyle="-.")
    plt.legend(loc='best', fontsize=8)
    plt.xlabel("group")
    plt.title(f"{dataset}_{model}_{method_path}_{target_entry}_{metric}")
    plt.savefig(f"analysis/{dataset}_{model}_{method_path}_{target_entry}_{metric}_{mean_f1}.pdf")
