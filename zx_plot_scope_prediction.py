import numpy as np
import matplotlib.pyplot as plt
from args import *
from sklearn import metrics
from utils import *

save_path = ""


def plot_multi_curve(shape, file_name, target_dim,terms):
    target = "Power" if target_dim == 10 else "CPU Usage"
    plt.cla()
    plt.figure(figsize=(8 * shape, 6 * shape))
    # for stgat
    predicts = np.load(f"{save_path}/best_predict_{target_dim}_hour_1.npy")[:,target_dim,:].squeeze()
    inner_gts = np.load(f"{save_path}/inner_gts_{target_dim}_hour_1.npy")[:,target_dim,:].squeeze()
    # for arima
    # predicts = np.load(f"output/ZX/analysis/best_predict_{target_dim}_CPU Usage_{12}_hour.npy").squeeze()
    # inner_gts = np.load(f"output/ZX/analysis/inner_gts_{target_dim}_CPU Usage_{12}_hour.npy").squeeze()

    predicts,inner_gts = get_std_data(predicts, inner_gts)
    predicts = predicts[:, terms]
    inner_gts_p = inner_gts[:, terms]
    predict_mean = predicts.mean(axis=1)
    inner_gt_mean = inner_gts.mean(axis=1)
    predict_max = predicts.max(axis=1)
    inner_gt_max = inner_gts.max(axis=1)
    predict_min = predicts.min(axis=1)
    inner_gt_min = inner_gts.min(axis=1)
    x = range(len(predict_mean))
    plt.plot(x, inner_gt_mean, "b", label=f"{target}")
    plt.plot(x, predict_mean, "r", label=f"predict_{target}")
    plt.fill_between(x, predict_min, predict_max, facecolor='r', alpha=0.2)
    plt.fill_between(x, inner_gt_min, inner_gt_max, facecolor='b', alpha=0.2)
    plt.legend()
    plt.savefig(f"analysis/{file_name}_{target_dim}.pdf")
    print(f"save files analysis/{file_name}_{target_dim}.pdf")
    scope_mse = up_down_mse_for_hour(predicts, inner_gts)
    point_mse = metrics.mean_squared_error(predicts.ravel(), inner_gts_p.ravel())
    print(f"point_mse: {point_mse} scope_mse: {scope_mse}")


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "ZX"
    args.group = "computer-b0503-01"
    print(args)
    global save_path
    # dir = "multi_dimensions"
    dir = "multi_dimensions_continous_train"
    args.condition_control = False
    args.save_dir = f"{dir}_{args.condition_control}"
    save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
    print(save_path)
    target_dim = 0
    plot_multi_curve(2, f"zx_stgat_hour_{dir}_{args.condition_control}", target_dim, list(range(0,12,1)))
    # plot_multi_curve(2, f"zx_arima_hour", target_dim, list(range(0, 12,1)))

    # np.array(list(range(0,12)),dtype=np.int32)


if __name__ == '__main__':
    main()
