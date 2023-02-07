import os
import numpy as np
import pandas as pd

# dataset = "SMD"
# end = ".txt"
# prefix = "machine-"
# method_type = "baseline_stgat"
# # model_type = "wo_norm"
# model_type = "norm"
# save_dir = f"{method_type}_{model_type}"
# dir_path = f"data/SMD/train"
# bs = 128
# for file_name in os.listdir(dir_path):
#     if file_name.endswith(end):
#         group = file_name.split(end)[0]
#         group = group.split(prefix)[1] # for smd
#         cmd = f"python main_stgat_v2.py --dataset {dataset} --group {group} --save_dir {save_dir} " \
#               f"--cuda_device 2 --batch {bs}"
#         print(cmd)
#         os.system(cmd)

# # dataset = "SMAP"
# dataset = "MSL"
# method_type = "baseline_stgat"
# # model_type = "wo_norm"
# model_type = "norm"
# save_dir = f"{method_type}_{model_type}"
# open_maml = False
# using_labeled_val = False
# bs = 128
# cuda = "1"
# csv_path = f"data/SMAP/{dataset.lower()}_train_md.csv"
# meta_data = pd.read_csv(csv_path, sep=",", index_col="chan_id")
# for group in meta_data.index:
#     cmd = f"python main_stgat_v2.py --dataset {dataset} --group {group} --save_dir {save_dir}" \
#           f" --cuda_device {cuda}  --batch {bs}"
#     print(cmd)
#     os.system(cmd)

# dataset = "WADI"
# method_type = "baseline_gat"
# # model_type = "wo_norm"
# model_type = "norm"
# save_dir = f"{method_type}_{model_type}"
# open_maml = False
# using_labeled_val = False
# bs = 32
# cuda = "1"
# for i in range(5):
#     group = "A1"
#     cmd = f"python maml_for_ad.py --dataset {dataset} --group {group} --save_dir {save_dir}_{i}" \
#           f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
#           f" --cuda_device {cuda}  --bs {bs}"
#     print(cmd)
#     os.system(cmd)


# dataset = "SWAT"
# method_type = "baseline_stgat"
# model_type = "norm"
# save_dir = f"{method_type}_{model_type}"
# open_maml = False
# using_labeled_val = False
# bs = 64
# cuda = "0"
# group = "A1_A2"
# cmd = f"python main_stgat_v2.py --dataset {dataset} --group {group} --save_dir {save_dir} " \
#           f"--cuda_device 2 --batch {bs}"
# print(cmd)
# os.system(cmd)
# group = "A4_A5"
# cmd = f"python main_stgat_v2.py --dataset {dataset} --group {group} --save_dir {save_dir} " \
#           f"--cuda_device 2 --batch {bs}"
# print(cmd)
# os.system(cmd)


dataset = "WT"
method_type = "baseline_stgat"
model_type = "norm"
save_dir = f"{method_type}_{model_type}"
bs = 64
cuda = "0"
group = "WT03"
cmd = f"python main_stgat_v2.py --dataset {dataset} --group {group} --save_dir {save_dir} " \
          f"--cuda_device 2 --batch {bs}"
print(cmd)
os.system(cmd)
group = "WT23"
cmd = f"python main_stgat_v2.py --dataset {dataset} --group {group} --save_dir {save_dir} " \
          f"--cuda_device 2 --batch {bs}"
print(cmd)
os.system(cmd)