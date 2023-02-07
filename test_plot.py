import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from args import *
from tqdm import tqdm

save_path = ""


# class Att:
#     def __init__(self,start,end,weight):
#         self.start = start
#         self.end = end
#         self.weight = weight


# def main_test():
#     x = np.linspace(0, 1, 500)
#     y = np.sin(3 * np.pi * x) * np.exp(-4 * x) + np.cos(x)
#     fig, ax = plt.subplots()
#     plt.plot(x, y)
#     # plt.fill_between(x,0,y,facecolor = 'green', alpha = 0.3)
#     plt.fill_between([0.1, 0.3], 0.2, 0.9, facecolor='pink', alpha=0.9)
#     plt.savefig("x.pdf")


# def plot_curve(data):
#     for i in data:
#         x = range(0, len(i))
#         plt.plot(x, i)
#         plt.savefig(f"GT_{i}.pdf")


def plot_double_curve(data1, data2, name):
    length = len(data1[0])
    for i in tqdm(range(length)):
        x = range(len(data1))
        y1 = data1[:, i]
        y2 = data2[:, i]
        # plt.figure()
        plt.cla()
        plt.plot(x, y1, "b", label="ground_truth")
        plt.plot(x, y2, "y", label="prediction")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f"{save_path}/{name}_{i}.pdf")


# def plot_curve_and_attention(data1, attention:Att):
#     length = len(data1)
#     for i in length:
#         x = range(0, len(data1[i]))
#         y1 = data1[i]
#         y_max = np.max(y1)
#         plt.plot(x, y1,"b")
#         plt.fill_between([attention.start, attention.end], 0, y_max, facecolor='pink', alpha=0.9*attention.weight)
#         plt.savefig(f"GT_Attention_{i}.pdf")


def plot_curve_and_attention_v2(inner_gt, inter_gt, attention):
    length = len(inner_gt[0])
    attentions = []
    index_list, lens_list = find_positive_segment(inter_gt)
    for i in tqdm(range(length)):
        x = range(len(inner_gt))
        y1 = inner_gt[:, i]
        y_max = np.max(y1)
        # plt.figure()
        plt.cla()
        plt.plot(x, y1, "b", label="ground_truth")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        temp = []
        for j in range(len(index_list)):
            start = index_list[j]
            end = index_list[j] + lens_list[j]
            weight = np.mean(attention[start:end, i])  # todo 检查为何数据维度不对
            temp.append(weight)
            plt.fill_between([start, end], 0, y_max, facecolor='pink', alpha=0.1 * weight)
        attentions.append(np.array(temp))
        plt.savefig(f"{save_path}/GT_Attention_{i}.pdf")
    return np.array(attentions)  # 返回每一个异常事件时，对于每一个维度的注意力权重


def find_positive_segment(data):
    dataset_len = int(len(data))
    index_list = []
    lens_list = []
    find = 0
    count = 0
    for i in range(len(data)):
        if int(data[i]) == 1:
            if find == 0:
                index_list.append(i)
            find = 1
            count += 1
        elif find == 1:
            find = 0
            lens_list.append(count)
            count = 0
    return index_list, lens_list


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    global save_path
    save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
    print(save_path)
    # recons = np.load(f"{save_path}/best_recons.npy")
    recons = np.load(f"{save_path}/best_predict.npy")
    # enhanced_attentions = np.load(f"{save_path}/best_attentions_Ture_data_enhancement.npy")
    # un_enhanced_attentions = np.load(f"{save_path}/best_attentions_False_data_enhancement.npy")
    # attentions = np.load(f"{save_path}/best_attentions_{args.open_maml}_data_enhancement.npy")  # todo check 这个注意力获取的可能有问题
    inner_gt = np.load(f"{save_path}/inner_gts.npy")
    # inter_gt = np.load(f"{save_path}/inter_gts.npy")
    plot_double_curve(inner_gt, recons, "gt_recons")
    # plot_double_curve(inner_gt,attentions)
    # plot_double_curve(preprocessing.normalize(np.abs(recons - inner_gt)),
    #                   attentions)  # 对比每个维度的重构误差和注意力，来评价注意力的好坏 # 考虑将重构误差（纵向？维度间）标准化
    # new_attentions = plot_curve_and_attention_v2(inner_gt, inter_gt, attentions)
    # print(new_attentions.argmax(axis=0))


if __name__ == '__main__':
    main()

# 目的1：DMAD注意力层对于异常数据的可视化 （绘制真实曲线和注意力权重热力图）
# 目的1.1：用于评价哪个维度的注意力应该更高，也就是对比维度的根因和注意力，要和前边实证研究的哪个维度更重要对上(SMD1-4没对上)，也要相比于未进行数据增强前，重要维度的注意力要更高
# 目的1.2：也可以对比维度的重构误差和注意力


# 目的2：绘制真实曲线、预测/重构曲线、重构误差曲线


# 读入数据，描绘真实曲线


# 读入预测数据，描绘预测/重构曲线


# 描绘重构误差


# 思考：是否耦合进一场检测过程？否，没必要。目的1可以耦合，目的2要解耦。


# todo：导出attention层数据，导出预测/重构数据 done
# todo：绘制目的1和目的2 
# todo：构建ARIMA
