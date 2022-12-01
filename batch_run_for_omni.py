import os
import numpy as np
import pandas as pd

# for smd
# dataset = "SMD"
# end = ".txt"
# prefix = "machine-"
# # dataset = "SMAP"
# # end = ".npy"
# dir_path = f"data/{dataset}/train"
# for file_name in os.listdir(dir_path):
#     if file_name.endswith(end):
#         group = file_name.split(end)[0]
#         group = group.split(prefix)[1] # for smd
#         os.system(f"python main_omni.py --dataset {dataset} --group {group} --cuda_device 0")

# for SMAP/MSL
dataset = "SMAP"
# dataset = "MSL"
csv_path = f"data/{dataset}/{dataset.lower()}_train_md.csv"
meta_data = pd.read_csv(csv_path, sep=",", index_col="chan_id")
for group in meta_data.index:
    os.system(f"python main_omni.py --dataset {dataset} --group {group} --cuda_device 1")