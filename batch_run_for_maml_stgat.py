import os
import numpy as np
import pandas as pd

method_type = "maml_stgat_un"
model_type = "norm"
# save_dir = f"{method_type}_{model_type}"
save_dir = "baseline_stgat"
open_maml = False
using_labeled_val = False
bs = 128
epoch = 10
cuda = "2"

# dataset = "SMD"
# end = ".txt"
# prefix = "machine-"
# dir_path = f"data/{dataset}/train"
# for file_name in os.listdir(dir_path):
#     if file_name.endswith(end):
#         group = file_name.split(end)[0]
#         group = group.split(prefix)[1]  # for smd
#         for i in range(1):
#             cmd = f"python main_stgat_v2.py --dataset {dataset} --group {group} --save_dir {save_dir}_{i}" \
#                   f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
#                   f" --cuda_device {cuda}  --bs {bs} --batch {bs} --epoch {epoch} --epochs {epoch}"
#             print(cmd)
#             os.system(cmd)

# dataset = "SMAP"
# dataset = "MSL"
# csv_path = f"data/SMAP/{dataset.lower()}_train_md.csv"
# meta_data = pd.read_csv(csv_path, sep=",", index_col="chan_id")
# for group in meta_data.index:
#     for i in range(1):
#         cmd = f"python main_stgat_v2.py --dataset {dataset} --group {group} --save_dir {save_dir}_{i}" \
#               f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
#               f" --cuda_device {cuda}  --bs {bs} --batch {bs} --epoch {epoch} --epochs {epoch}"
#         print(cmd)
#         os.system(cmd)

# dataset = "WADI"
# for i in range(5):
#     group = "A2"
#     cmd = f"python main_stgat_v2.py --dataset {dataset} --group {group} --save_dir {save_dir}_{i}" \
#               f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
#               f" --cuda_device {cuda}  --bs {bs} --batch {bs} --epoch {epoch} --epochs {epoch}"
#     print(cmd)
#     os.system(cmd)

dataset = "SWAT"
groups = ["A1_A2", "A4_A5"]
for group in groups:
    for i in range(5):
        cmd = f"python main_stgat_v2.py --dataset {dataset} --group {group} --save_dir {save_dir}_{i}" \
              f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
              f" --cuda_device {cuda}  --bs {bs} --batch {bs} --epoch {epoch} --epochs {epoch}"
        print(cmd)
        os.system(cmd)


# dataset = "BATADAL"
# for i in range(1):
#     group = "A2"
#     cmd = f"python main_stgat_v2.py --dataset {dataset} --group {group} --save_dir {save_dir}_{i}" \
#           f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
#           f" --cuda_device {cuda}  --bs {bs} --batch {bs} --epoch {epoch} --epochs {epoch}"
#     print(cmd)
#     os.system(cmd)
