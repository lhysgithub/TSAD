import os
import numpy as np
import pandas as pd

dataset = "SMD"
end = ".txt"
prefix = "machine-"
dir_path = f"data/{dataset}/train"
for file_name in os.listdir(dir_path):
    if file_name.endswith(end):
        group = file_name.split(end)[0]
        group = group.split(prefix)[1] # for smd
        cmd = f"python maml_for_ad_run.py --dataset {dataset} --group {group} --cuda_device 3"
        print(cmd)
        os.system(cmd)



# # for SMAP/MSL
# dataset = "SMAP"
# # dataset = "MSL"
# csv_path = f"data/{dataset}/{dataset.lower()}_train_md.csv"
# meta_data = pd.read_csv(csv_path, sep=",", index_col="chan_id")
# for group in meta_data.index:
#     cmd = f"python maml_for_ad.py --dataset {dataset} --group {group} --cuda_device 2"
#     print(cmd)
#     os.system(cmd)
