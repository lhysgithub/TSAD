import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from utils import *


def get_data(dataset, group, metric):
    f1s = []
    old_f1 = 0
    for j in range(1, 101):
        save_dir = f"ROI_{j}"
        file_name2 = f"output/{dataset}" + "/" + group + f"/{save_dir}/summary.txt"
        try:
            f1 = get_key_for_maml(file_name2, metric)
            if f1 < 0.1:
                f1 = old_f1
            f1s.append(f1)
            old_f1 = f1
        except Exception as e:
            print(f"error process in {file_name2}")
            continue
    return f1s


if __name__ == '__main__':
    dataset = "SWAT"
    group = "A4_A5"
    metric = "f1"
    f1s = get_data(dataset, group, metric)
    plt.figure(dpi=300, figsize=(30, 8))
    plt.plot(np.array(range(1, len(f1s)+1))*0.01, f1s, label=f"{metric}")
    plt.legend(loc='best', fontsize=8)
    plt.xlabel("group")
    name = f"ROI_{dataset}_{group}_{metric}"
    plt.title(name)
    plt.savefig(f"analysis/{name}.pdf")
