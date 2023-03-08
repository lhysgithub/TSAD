import numpy as np
import matplotlib.pyplot as plt
from args import *


save_path = ""


def plot_multi_curve(shape, file_name, target_dim):
    target = "Power" if target_dim == 10 else "CPU Usage"
    plt.cla()
    plt.figure(figsize=(8 * shape, 6 * shape))
    predicts = np.load(f"{save_path}/best_predict_{target_dim}_hour.npy")[:,target_dim,:].squeeze()
    inner_gts = np.load(f"{save_path}/inner_gts_{target_dim}_hour.npy")[:,target_dim,:].squeeze()
    predict_mean = predicts.mean(axis=1)
    inner_gt_mean = inner_gts.mean(axis=1)
    predict_max = predicts.max(axis=1)
    inner_gt_max = inner_gts.max(axis=1)
    predict_min = predicts.min(axis=1)
    inner_gt_min = inner_gts.min(axis=1)
    x = range(len(predict_mean))
    plt.plot(x, inner_gt_mean, "b", label=f"{target}")
    # plt.plot(x, inner_gt_max, "b", label=f"{target}_max", alpha=0.8)
    # plt.plot(x, inner_gt_min, "b", label=f"{target}_min", alpha=0.5)

    plt.plot(x, predict_mean, "r", label=f"predict_{target}")
    # plt.plot(x, predict_max, "r", label=f"predict_{target}_max", alpha=0.8)
    # plt.plot(x, predict_min, "r", label=f"predict_{target}_min", alpha=0.5)
    plt.fill_between(x, predict_min, predict_max, facecolor='r', alpha=0.2)
    plt.fill_between(x, inner_gt_min, inner_gt_max, facecolor='b', alpha=0.2)
    plt.legend()
    plt.savefig(f"analysis/{file_name}_{target_dim}.pdf")


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "ZX"
    args.group = "computer-b0503-01"
    print(args)
    global save_path
    save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
    print(save_path)
    target_dim = 0
    plot_multi_curve(5, "zx_stgat_day_infer_hour_scope_condition_control", target_dim)


if __name__ == '__main__':
    main()
