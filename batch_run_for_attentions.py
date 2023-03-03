import os

open_maml = True
using_labeled_val = True # True False
save_dir = f"{open_maml}_DC_{using_labeled_val}_Semi"
bs = 128
epoch = 10
cuda = "1"

dataset = "SMD"
end = ".txt"
prefix = "machine-"
dir_path = f"data/{dataset}/train"
for file_name in os.listdir(dir_path):
    if file_name.endswith(end):
        group = file_name.split(end)[0]
        group = group.split(prefix)[1]  # for smd
        for i in range(5):
            cmd = f"python maml_for_ad_v2.py --dataset {dataset} --group {group} --save_dir {save_dir}" \
                  f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} " \
                  f" --cuda_device {cuda}  --bs {bs} --batch {bs} --epoch {epoch} --epochs {epoch} "
            print(cmd)
            os.system(cmd)