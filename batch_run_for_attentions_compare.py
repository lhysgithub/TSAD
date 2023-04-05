import os

# open_maml = True
# using_labeled_val = True # True False

bs = 128
epoch = 20
cuda = "1"
dataset = "SMD"
group = "1-4"
for j in range(3):
    for open_maml in [True,False]:
        for using_labeled_val in [True,False]:
            save_dir = f"{open_maml}_DC_{using_labeled_val}_Semi_0319_{j}"
            for i in range(5):
                cmd = f"python maml_for_ad_v4.py --dataset {dataset} --group {group} --save_dir {save_dir}" \
                      f" --open_maml {open_maml} --using_labeled_val {using_labeled_val} " \
                      f" --cuda_device {cuda}  --bs {bs} --batch {bs} --epoch {epoch} --epochs {epoch} "
                print(cmd)
                os.system(cmd)