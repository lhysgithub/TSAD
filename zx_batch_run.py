import os

gaps = [1,2,3,4]
# save_dir = "few_dimensions"
# save_dir = "multi_dimensions"
save_dir = "multi_dimensions_continous_train"

# for gap_ in gaps:
#     cmd = f"python zx_stgat_cpu_model.py --save_dir {save_dir}_{gap_} --pre_gap {gap_} --condition_control True"
#     print(cmd)
#     os.system(cmd)
cmd = f"python zx_stgat_cpu_model.py --save_dir {save_dir}_{True} --condition_control True --pre_gap 1"
print(cmd)
os.system(cmd)

cmd = f"python zx_stgat_cpu_model.py --save_dir {save_dir}_{False} --condition_control False --pre_gap 1"
print(cmd)
os.system(cmd)