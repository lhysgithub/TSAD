import os
import numpy as np
import pandas as pd

open_maml = True
using_labeled_val = True
retrain = True
model_type = "norm"
bs = 128
epoch = 30
cuda = "1"

# dataset = "SMD"
# group = "2-5"

#dataset = "SWAT"
#group = "A4_A5"
#group = "A1_A2"

dataset = "BATADAL"
group = "A2"

for j in range(1, 101):
    save_dir = f"ROI_v2_{j}"
    for i in range(3):
        cmd = f"python maml_for_ad_v2.py --val_split {j*0.01} --dataset {dataset} --group {group} --save_dir {save_dir}" \
              f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} --norm_model {model_type}" \
              f" --cuda_device {cuda}  --bs {bs} --batch {bs} --epoch {epoch} --epochs {epoch} --retrain {retrain}"
        print(cmd)
        os.system(cmd)
