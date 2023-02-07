import os
import numpy as np
import pandas as pd

# dataset = "SMD"
# end = ".txt"
# prefix = "machine-"
# method_type = "maml_un_v2"
# model_type = "norm"  # "wo_norm"
# save_dir = f"{method_type}_{model_type}"
# open_maml = True
# using_labeled_val = False
# bs = 128
# cuda = "2"
# dir_path = f"data/{dataset}/train"
# for i in range(3):
#     for file_name in os.listdir(dir_path):
#         if file_name.endswith(end):
#             group = file_name.split(end)[0]
#             group = group.split(prefix)[1]  # for smd
#             cmd = f"python maml_for_ad_v2.py --dataset {dataset} --group {group} --save_dir {save_dir}_{i}" \
#                   f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
#                   f" --cuda_device {cuda}  --bs {bs}"
#             print(cmd)
#             os.system(cmd)

# # dataset = "SMAP"
# dataset = "MSL"
# method_type = "maml_un_v2"
# model_type = "norm" # "wo_norm"
# save_dir = f"{method_type}_{model_type}"
# open_maml = True
# using_labeled_val = False
# bs = 128
# cuda = "1"
# csv_path = f"data/SMAP/{dataset.lower()}_train_md.csv"
# meta_data = pd.read_csv(csv_path, sep=",", index_col="chan_id")
# for i in range(3):
#     for group in meta_data.index:
#         cmd = f"python maml_for_ad_v2.py --dataset {dataset} --group {group} --save_dir {save_dir}_{i}" \
#               f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
#               f" --cuda_device {cuda}  --bs {bs}"
#         print(cmd)
#         os.system(cmd)

# dataset = "WADI"
# method_type = "maml_un"
# model_type = "norm" # "wo_norm"
# save_dir = f"{method_type}_{model_type}"
# open_maml = True
# using_labeled_val = False
# bs = 32
# cuda = "0"
# for i in range(5):
#     group = "A2"
#     cmd = f"python maml_for_ad.py --dataset {dataset} --group {group} --save_dir {save_dir}_{i}" \
#           f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
#           f" --cuda_device {cuda}  --bs {bs}"
#     print(cmd)
#     os.system(cmd)

# dataset = "SWAT"
# method_type = "maml_un"
# model_type = "norm"
# save_dir = f"{method_type}_{model_type}"
# open_maml = True
# using_labeled_val = False
# bs = 64
# cuda = "0"
# for i in range(5):
# group = "A1_A2"
# cmd = f"python maml_for_ad.py --dataset {dataset} --group {group} --save_dir {save_dir}_{i}" \
#       f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
#       f" --cuda_device {cuda}  --bs {bs}"
# print(cmd)
# os.system(cmd)
# group = "A4_A5"
# cmd = f"python maml_for_ad.py --dataset {dataset} --group {group} --save_dir {save_dir}_{i}" \
#       f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
#       f" --cuda_device {cuda}  --bs {bs}"
# print(cmd)
# os.system(cmd)

# dataset = "BATADAL"
# method_type = "maml_un"
# model_type = "norm"
# save_dir = f"{method_type}_{model_type}"
# open_maml = True
# using_labeled_val = False
# bs = 64
# cuda = "0"
# for i in range(3):
#     group = "A1_A2"
#     cmd = f"python maml_for_ad.py --dataset {dataset} --group {group} --save_dir {save_dir}_{i}" \
#           f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
#           f" --cuda_device {cuda}  --bs {bs}"
#     print(cmd)
#     os.system(cmd)