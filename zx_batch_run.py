import os

# gaps = [1,2,3,4]
# # save_dir = "few_dimensions"
# # save_dir = "multi_dimensions"
# save_dir = "multi_dimensions_continous_train"

# # for gap_ in gaps:
# #     cmd = f"python zx_stgat_cpu_model.py --save_dir {save_dir}_{gap_} --pre_gap {gap_} --condition_control True"
# #     print(cmd)
# #     os.system(cmd)
# cmd = f"python zx_stgat_cpu_model.py --save_dir {save_dir}_{True} --condition_control True --pre_gap 1"
# print(cmd)
# os.system(cmd)

# cmd = f"python zx_stgat_cpu_model.py --save_dir {save_dir}_{False} --condition_control False --pre_gap 1"
# print(cmd)
# os.system(cmd)

for slide_win in [288]:
    # for epoch in [1]:
# for slide_win in [24,48,96,288]:
    for epoch in [1,2,4,6,8]:
        cmd = f"python zx_stgat_cpu_model_v2.py --slide_win {slide_win} --epoch {epoch} --multi_entities_train True --condition_infer True --only_plot_untrain False" # condition_infer test
        # cmd = f"python zx_stgat_cpu_model_v2.py --slide_win {slide_win} --epoch {epoch} --multi_entities_train False --condition_infer False --only_plot_untrain True" # plot_untrain test
        print(cmd)
        os.system(cmd)