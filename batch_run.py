import os

dataset = "SMAP"
end = ".npy"

# dataset = "SMD"
# end = ".txt"
# prefix = "machine-"
dir_path = f"data/{dataset}/train"
for file_name in os.listdir(dir_path):
    if file_name.endswith(end):
        group = file_name.split(end)[0]
        # group = group.split(prefix)[1] # for smd
        os.system(f"python train.py --dataset {dataset} --group {group} --cuda_device 0")
