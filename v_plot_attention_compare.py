import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from args import *
from tqdm import tqdm
import pandas as pd
import seaborn as sns

save_path = ""

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

def get_attentions_in_abnormal(inter_gt, attention):
    index_list, lens_list = find_positive_segment(inter_gt)
    abnormal_attentions = []
    for j in range(len(index_list)):
        start = index_list[j]
        end = index_list[j] + lens_list[j]
        abnormal_attentions.append(attention[start:end])
    return abnormal_attentions


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
            plt.fill_between([start, end], 0, y_max, facecolor='pink', alpha=0.05 * weight)
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


def get_anormal_dimensions(args):
    anormal = []
    file_name = f"data/SMD/interpretation_label/machine-{args.group}.txt"
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            temp_list = line.split(":")[1].split(",")
            anormal.append([int(i) for i in temp_list])
    return anormal


def display_result(args):
    print(args)
    global save_path
    save_path = f"output/{args.dataset}/{args.group}/{args.open_maml}_DC_{args.using_labeled_val}_Semi"
    print(save_path)
    attentions = np.load(f"{save_path}/best_attentions_{args.open_maml}_data_enhancement_{args.using_labeled_val}_semi_all.npy")
    display_attention_inner_stds(attentions)
    attentions = attentions.mean(axis=1)
    inner_gt = np.load(f"{save_path}/inner_gts.npy")
    inter_gt = np.load(f"{save_path}/inter_gts.npy")
    anormals = get_anormal_dimensions(args)
    display_inner_gt(args, inner_gt, inter_gt)
    # 对比每个维度的重构误差和注意力，来评价注意力的好坏 # 考虑将重构误差（纵向？维度间）标准化
    # new_attentions = plot_curve_and_attention_v2(inner_gt, inter_gt, attentions)
    # print(new_attentions.argmax(axis=0))

    display_information(args, inner_gt, inter_gt, attentions, anormals)


def display_attention_cross_std(inner_gt, attentions):
    overall_attention_on_dims = attentions.T.mean(axis=1)
    inner_gt_on_dims_std = inner_gt.T.std(axis=1)
    cross_std = np.multiply(inner_gt_on_dims_std, overall_attention_on_dims).sum()
    return cross_std


def display_attention_inner_stds(attentions):
    # attentions_inner_std_mean = attentions.std(axis=2).T.mean(axis=1).mean(axis=0)
    attentions_inner_std_mean = attentions.std(axis=2).mean()
    print(f"attentions_inner_std_mean {attentions_inner_std_mean}")


def display_attention_topk_in_anormal(attentions, anormals, k):
    overall_attention_on_dims = attentions.T.mean(axis=1)
    ind = np.argpartition(overall_attention_on_dims, -k)[-k:]
    ind = ind[np.argsort(-overall_attention_on_dims[ind])]
    ind_1 = [i + 1 for i in ind]
    elements_length_anormals = 0
    elements_anormals = []
    elements_length_in_topk = 0
    elements_bingo_length_in_topk = 0
    for i in anormals:
        elements_length_anormals = elements_length_anormals + len(i)
        elements_anormals = elements_anormals + i
    for i in ind_1:
        for j in anormals:
            if i in j:
                elements_length_in_topk = elements_length_in_topk + 1
    for i in ind_1:
        if i in elements_anormals:
            elements_bingo_length_in_topk = elements_bingo_length_in_topk + 1
    return elements_length_in_topk*1.0/elements_length_anormals
    # return elements_bingo_length_in_topk*1.0/len(ind_1)


def find_anormal_attentions_length(inter_gt, attentions):
    abnormal_attentions = get_attentions_in_abnormal(inter_gt, attentions)
    length = 0
    for i in range(len(abnormal_attentions)):
        attentions_ = abnormal_attentions[i]
        length = length + len(attentions_)
    return length


def display_inner_gt(args, inner_gt, inter_gt):
    k=1
    plt.figure(figsize=(16*k, 9*k))
    for j in range(len(inner_gt.T)):
        i = inner_gt.T[j]
        plt.cla()
        plt.plot(range(len(i)), i, label=f"{j}")
        index_list, lens_list = find_positive_segment(inter_gt)
        min_ = i.min()
        max_ = i.max()
        for k in range(len(index_list)):
            start = index_list[k]
            end = index_list[k] + lens_list[k]
            plt.fill_between([start, end], min_, max_, facecolor='red', alpha=0.5)
        plt.savefig(f"analysis/{args.dataset}_{args.group}_{j}.pdf")



def display_in_anormal_information(args, inner_gt, inter_gt, attentions, anormals):
    abnormal_attentions = get_attentions_in_abnormal(inter_gt, attentions)
    for i in range(1, len(abnormal_attentions)):
        attentions_ = abnormal_attentions[i]
        overall_attention_on_dims = attentions_.T.mean(axis=1)
        print(f"series length: {len(attentions_)}")
        print(f"DC: {args.open_maml} semi-supervise: {args.using_labeled_val}")
        print(f"attentions: {overall_attention_on_dims}")
        print(f"mean: {overall_attention_on_dims.mean()} std: {overall_attention_on_dims.std()}")
        k = 10
        ind = np.argpartition(overall_attention_on_dims, -k)[-k:]
        ind = ind[np.argsort(-overall_attention_on_dims[ind])]
        ind_1 = [i+1 for i in ind]
        print(f"top k: {overall_attention_on_dims[ind]}")
        print(f"top k index: {ind_1}")
        bingo_ratio = display_attention_topk_in_anormal(attentions_, [anormals[i]], k)
        print(f"top k bingo_ratio: {bingo_ratio}")
        print()
        break


def display_information(args, inner_gt, inter_gt, attentions, anormals):
    overall_attention_on_dims = attentions.T.mean(axis=1)
    length = find_anormal_attentions_length(inter_gt, attentions)
    cross_std = display_attention_cross_std(inner_gt, attentions)
    print(f"series length: {len(attentions)}")
    print(f"anomaly length: {length}")
    print(f"DC: {args.open_maml} semi-supervise: {args.using_labeled_val}")
    print(f"attentions: {overall_attention_on_dims}")
    print(f"mean: {overall_attention_on_dims.mean()} std: {overall_attention_on_dims.std()} cross_std {cross_std}")
    k = 15
    ind = np.argpartition(overall_attention_on_dims, -k)[-k:]
    ind = ind[np.argsort(-overall_attention_on_dims[ind])]
    ind_1 = [i + 1 for i in ind]
    print(f"top k: {overall_attention_on_dims[ind]}")
    print(f"top k index: {ind_1}")
    bingo_ratio = display_attention_topk_in_anormal(attentions, anormals, k)
    print(f"top k bingo_ratio: {bingo_ratio}")
    print()


def get_attentions(args):
    global save_path
    save_path = f"output/{args.dataset}/{args.group}/{args.open_maml}_DC_{args.using_labeled_val}_Semi"
    attentions = np.load(f"{save_path}/best_attentions_{args.open_maml}_data_enhancement_{args.using_labeled_val}_semi_all.npy")
    display_attention_inner_stds(attentions)
    attentions = attentions.mean(axis=1)
    inner_gt = np.load(f"{save_path}/inner_gts.npy")
    inter_gt = np.load(f"{save_path}/inter_gts.npy")
    return attentions, inner_gt, inter_gt


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.open_maml = False
    args.using_labeled_val = False  # True False
    wosemdc_attentions, _, _ = get_attentions(args)
    args.open_maml = True
    args.using_labeled_val = False  # True False
    wosem_attentions, _, _ = get_attentions(args)
    args.open_maml = False
    args.using_labeled_val = True  # True False
    wodc_attentions, _, _ = get_attentions(args)
    args.open_maml = True
    args.using_labeled_val = True  # True False
    semdc_attentions, inner_gt, inter_gt = get_attentions(args)
    anormals = get_anormal_dimensions(args)
    anormals_marix = np.zeros((12, 38))
    for i in range(12):
        for j in anormals[i]:
            anormals_marix[i,j-1] = 1.0
    attention_compare_matrix = []
    attention_compare_matrix.append(wosemdc_attentions.T.mean(axis=1))
    attention_compare_matrix.append(wosem_attentions.T.mean(axis=1))
    attention_compare_matrix.append(wodc_attentions.T.mean(axis=1))
    attention_compare_matrix.append(semdc_attentions.T.mean(axis=1))
    attention_compare_matrix.append(anormals_marix.T.mean(axis=1))
    attention_compare_matrix = np.array(attention_compare_matrix)
    center = np.mean(attention_compare_matrix[:4])
    max_1 = np.max(attention_compare_matrix[:4])
    max_2 = np.max(attention_compare_matrix[4])
    attention_compare_matrix[4] = (attention_compare_matrix[4]/max_2) * max_1
    data_pd = pd.DataFrame(attention_compare_matrix.T, index=range(1, 39) ,columns=["w/o Sem & DC", "w/o Sem", "w/o DC", "SemDC",
                                                                "Abnormal Ratio"])
    print(f"12-1: {data_pd.iloc[12,0]} 12-4: {data_pd.iloc[12,3]} 12-5: {data_pd.iloc[12,4]}")
    print(f"27-1: {data_pd.iloc[27, 0]} 27-4: {data_pd.iloc[27, 3]} 27-5: {data_pd.iloc[27, 4]}")

    plt.figure(figsize=(6, 8))
    # data_pd = pd.DataFrame(attention_compare_matrix.T[:,:4], columns=["w/o Sem & DC", "w/o Sem", "w/o DC", "SemDC"])
    p = sns.heatmap(data_pd,  cmap="RdBu_r",
                    cbar_kws={"label": f"Attention"}, center=center, #vmin=0, vmax=1,
                    square=False)#yticklabels=ylabes,
    p.set_ylabel("Index")
    title = "Attention Visual Comparison"
    plt.title(title)
    plt.xticks(rotation=30)
    plt.savefig(f"analysis/{title}.pdf")


if __name__ == '__main__':
    main()
