import os

bs = 128
epoch = 10
cuda = "1"
model = "stgat" # mtad_gat
# model = "mtad_gat"

dataset = "SMD"
group = "1-4"
for i in range(38,39):
    for j in range(2,4):
        save_dir = f"{model}_removed_variate_{i}_{j}"
        cmd = f"python baseline_{model}_subset.py --dataset {dataset} --group {group} --save_dir {save_dir}" \
            f" --removed_variate {i}"\
            f" --cuda_device {cuda}  --bs {bs} --batch {bs} --epoch {epoch} --epochs {epoch} "
        print(cmd)
        os.system(cmd)

dataset = "SMAP"
group = "A-7"
for i in range(25,26):
    for j in range(2,4):
        save_dir = f"{model}_removed_variate_{i}_{j}"
        cmd = f"python baseline_{model}_subset.py --dataset {dataset} --group {group} --save_dir {save_dir}" \
              f" --removed_variate {i}"\
            f" --cuda_device {cuda}  --bs {bs} --batch {bs} --epoch {epoch} --epochs {epoch} "
        print(cmd)
        os.system(cmd)

dataset = "MSL"
group = "T-8"
for i in range(50,51):
    for j in range(2,4):
        save_dir = f"{model}_removed_variate_{i}_{j}"
        cmd = f"python baseline_{model}_subset.py --dataset {dataset} --group {group} --save_dir {save_dir}" \
              f" --removed_variate {i}"\
            f" --cuda_device {cuda}  --bs {bs} --batch {bs} --epoch {epoch} --epochs {epoch} "
        print(cmd)
        os.system(cmd)