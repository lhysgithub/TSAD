import os
import numpy as np
import pandas as pd

# dataset = "SMD"
# end = ".txt"
# prefix = "machine-"
# dir_path = f"data/{dataset}/train"
# for file_name in os.listdir(dir_path):
#     if file_name.endswith(end):
#         group = file_name.split(end)[0]
#         group = group.split(prefix)[1] # for smd
#         cmd = f"python maml_for_ad_val.py --dataset {dataset} --group {group} --cuda_device 1"
#         print(cmd)
#         os.system(cmd)



# # for SMAP/MSL
# dataset = "SMAP"
dataset = "MSL"
save_dir = "maml_val" # "maml_un"
csv_path = f"data/SMAP/{dataset.lower()}_train_md.csv"
meta_data = pd.read_csv(csv_path, sep=",", index_col="chan_id")
for group in meta_data.index:
    cmd = f"python maml_for_ad.py --dataset {dataset} --group {group} --save_dir {save_dir} " \
          f"--using_labeled_val True  --cuda_device 2"
    print(cmd)
    os.system(cmd)
