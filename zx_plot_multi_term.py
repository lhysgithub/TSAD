import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from args import *


save_path = ""


def plot_multi_curve(shape, file_name, i):  # i = target_dim
    target = "Power" if i == 10 else "CPU Usage"
    plt.cla()
    plt.figure(figsize=(8 * shape, 6 * shape))
    for j in range(1, 13):
        # recons = np.load(f"{save_path}/best_predict_{j}_term_{i}.npy")
        # inner_gt = np.load(f"{save_path}/inner_gts_{j}_term_{i}.npy")
        save_path = "output/ZX/analysis"
        recons = np.load(f"{save_path}/best_predict_{i}_{target}_{j}.npy")
        inner_gt = np.load(f"{save_path}/inner_gts_{i}_{target}_{j}.npy")
        plt.subplot(4, 3, j)
        x = range(len(recons))
        # y1 = recons[:, i]
        # y2 = inner_gt[:, i]
        y1 = recons[:, 0]
        y2 = inner_gt[:, 0]
        plt.plot(x, y2, "b", label=f"{target}")
        plt.plot(x, y1, "r", label=f"predict_{target}", alpha=0.5)  # , alpha=0.5
        plt.legend()
    plt.savefig(f"{save_path}/{file_name}_{i}.pdf")


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    global save_path
    save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
    print(save_path)
    target_dim = 10
    plot_multi_curve(5, "multi_term_arima", target_dim)


if __name__ == '__main__':
    main()


# todo：描绘不同预测距离下，真实值与预测值之间差距
