import os
import numpy as np
import pandas as pd

open_maml = True
using_labeled_val = True
retrain = True
model_type = "norm"
bs = 128
epoch = 10
cuda = "1"

R1 = np.array(range(1, 6)) * 0.2
R2 = R1
C = np.array(range(1, 6)) * 2
dataset = "SWAT"
group = "A4_A5"
for r1 in R1:
    for r2 in R2:
        for c in C:
            save_dir = f"grid_search_{r1}_{r2}_{c}"
            for i in range(3):
                cmd = f"python maml_for_ad_v2.py --r1 {r1} --r2 {r2} --confidence {c} --dataset {dataset} --group {group} --save_dir {save_dir}" \
                      f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
                      f" --cuda_device {cuda}  --bs {bs} --batch {bs} --epoch {epoch} --epochs {epoch} --retrain {retrain}"
                print(cmd)
                os.system(cmd)