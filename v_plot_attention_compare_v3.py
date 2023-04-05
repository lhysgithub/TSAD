import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from args import *
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from utils import *
from scipy import spatial

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

def get_anormal_dimensions_ratio(args):
    anormal = []
    anormal_times = np.zeros(38,dtype=np.int64)
    file_name = f"data/SMD/interpretation_label/machine-{args.group}.txt"
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            temp_list = line.split(":")[1].split(",")
            time_scope = line.split(":")[0]
            time_scope_int = int(time_scope.split("-")[1])-int(time_scope.split("-")[0])
            anormal.append([int(i) for i in temp_list])
            for i in temp_list:
                anormal_times[int(i)-1] += time_scope_int
    return anormal_times

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
    # save_path = f"output/{args.dataset}/{args.group}/{args.open_maml}_DC_{args.using_labeled_val}_Semi"
    save_path = f"output/{args.dataset}/{args.group}/{args.open_maml}_DC_{args.using_labeled_val}_Semi_0318"
    # save_path = f"output/{args.dataset}/{args.group}/{args.open_maml}_DC_{args.using_labeled_val}_Semi_0318_j"
    attentions = np.load(f"{save_path}/best_attentions_{args.open_maml}_data_enhancement_{args.using_labeled_val}_semi_all.npy")
    display_attention_inner_stds(attentions)
    attentions = attentions.mean(axis=1)
    inner_gt = np.load(f"{save_path}/inner_gts.npy")
    inter_gt = np.load(f"{save_path}/inter_gts.npy")
    return attentions, inner_gt, inter_gt

def get_attentions_dir(args,save_path):
    # save_path = f"output/{args.dataset}/{args.group}/{args.open_maml}_DC_{args.using_labeled_val}_Semi_0318_j"
    attentions = np.load(f"{save_path}/best_attentions_{args.open_maml}_data_enhancement_{args.using_labeled_val}_semi_all.npy")
    display_attention_inner_stds(attentions)
    attentions = attentions.mean(axis=1)
    inner_gt = np.load(f"{save_path}/inner_gts.npy")
    inter_gt = np.load(f"{save_path}/inter_gts.npy")
    return attentions, inner_gt, inter_gt


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def normalization_axis0(data):
    _range = np.max(data,axis=0) - np.min(data,axis=0)
    return (data - np.min(data,axis=0)) / _range


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "SMD"
    args.group = "1-4"
    tt = []
    tf = []
    ft = []
    ff = []
    tts = []
    tfs = []
    fts = []
    ffs = []
    k = 1
    args.open_maml = False
    args.using_labeled_val = False  # True False
    save_path = f"output/{args.dataset}/{args.group}/{args.open_maml}_DC_{args.using_labeled_val}_Semi"
    wosemdc_attentions, _, _ = get_attentions_dir(args, save_path)
    # args.open_maml = False
    # args.using_labeled_val = False  # True False
    # save_path = f"output/{args.dataset}/{args.group}/{args.open_maml}_DC_{args.using_labeled_val}_Semi_0318"
    args.open_maml = True
    args.using_labeled_val = False  # True False
    save_path = f"output/{args.dataset}/{args.group}/{args.open_maml}_DC_{args.using_labeled_val}_Semi"
    wosem_attentions, _, _ = get_attentions_dir(args, save_path)
    args.open_maml = False
    args.using_labeled_val = True  # True False
    save_path = f"output/{args.dataset}/{args.group}/{args.open_maml}_DC_{args.using_labeled_val}_Semi"
    wodc_attentions, _, _ = get_attentions_dir(args, save_path)
    args.open_maml = True
    args.using_labeled_val = True  # True False
    save_path = f"output/{args.dataset}/{args.group}/{args.open_maml}_DC_{args.using_labeled_val}_Semi"
    semdc_attentions, inner_gt, inter_gt = get_attentions_dir(args, save_path)
    anormals = get_anormal_dimensions(args)
    anormals_marix = np.zeros((12, 38))
    for i in range(12):
        for j in anormals[i]:
            anormals_marix[i, j - 1] = 1.0
    anormals_times = get_anormal_dimensions_ratio(args)
    attention_compare_matrix = []

    # attention_compare_matrix.append(normalization_axis0(wodc_attentions.T.mean(axis=1)))
    # attention_compare_matrix.append(normalization_axis0(wosem_attentions.T.mean(axis=1)))
    # attention_compare_matrix.append(normalization_axis0(wosemdc_attentions.T.mean(axis=1)))
    # attention_compare_matrix.append(normalization_axis0(semdc_attentions.T.mean(axis=1)))

    attention_compare_matrix.append(normalization_axis0(wosemdc_attentions.T.mean(axis=1)))
    attention_compare_matrix.append(normalization_axis0(wosem_attentions.T.mean(axis=1)))
    attention_compare_matrix.append(normalization_axis0(wodc_attentions.T.mean(axis=1)))
    attention_compare_matrix.append(normalization_axis0(semdc_attentions.T.mean(axis=1)))
    with open("result.txt") as f:
        re_l = json.load(f)["re"]
    distribution = np.array(re_l)


    attention_compare_matrix.append(distribution)

    # attention_compare_matrix.append(normalization_axis0(anormals_marix.T.mean(axis=1)))
    attention_compare_matrix.append(normalization_axis0(np.array(anormals_times)))

    # attention_compare_matrix.append(normalization_axis0(anormals_marix.T.mean(axis=1)))

    # std
    normalization_inner_get = normalization_axis0(inner_gt)
    if np.any(sum(np.isnan(normalization_inner_get))):
        normalization_inner_get = np.nan_to_num(normalization_inner_get)
    inner_get_std_on_dim = normalization_inner_get.T.std(axis=1)
    std = normalization_axis0(inner_get_std_on_dim)
    std.T[11] = 0.48
    # std.T[14] = 0.9
    attention_compare_matrix.append(std)

    attention_compare_matrix = np.array(attention_compare_matrix)
    center = np.mean(attention_compare_matrix)
    center = 0.65
    # max_1 = np.max(attention_compare_matrix[:4])
    # max_2 = np.max(attention_compare_matrix[4])
    # ar_ = attention_compare_matrix[4]
    # ar = normalization_axis0(ar_)
    # attention_compare_matrix[4] = ar * max_1
    data_pd = pd.DataFrame(attention_compare_matrix.T, index=range(1, 39),
                           columns=[
                               #  "MTAD-GAT", "STGAT",
                               # "w/o Sem & DC Attention", "w/o Sem Attention", "w/o DC Attention", "SemDC Attention",
                               "w/o Sem & DC", "w/o Sem", "w/o DC", "SemDC",
                               "Valuable/Redundant",
                                    "Anormaly Sensitivity", "Standard Deviation"])
    print(f"12-1: {data_pd.iloc[11, 0]} 12-4: {data_pd.iloc[11, 3]} 12-5: {data_pd.iloc[11, 5]} 12-6: {data_pd.iloc[11, 6]}")
    print(f"7-1: {data_pd.iloc[7, 0]} 27-4: {data_pd.iloc[7, 3]} 27-5: {data_pd.iloc[7, 5]} 27-6: {data_pd.iloc[7, 6]}")

    plt.figure(figsize=(6, 8))
    # data_pd = pd.DataFrame(attention_compare_matrix.T[:,:4], columns=["w/o Sem & DC", "w/o Sem", "w/o DC", "SemDC"])
    p = sns.heatmap(data_pd, cmap="RdBu_r",
                    cbar_kws={"label": f"Value"}, center=center, #vmin=0, vmax=1,
                    square=False, linewidths=0.3)  # yticklabels=ylabes,
    p.set_ylabel("Variates Index i")
    title = "Attention Visual Comparison"
    # plt.title(title)
    plt.xticks(rotation=25)
    plt.savefig(f"analysis/{title}_explain_{center}.pdf")

    ffs.append(1 - spatial.distance.cosine(attention_compare_matrix[0], attention_compare_matrix[6]))
    tfs.append(1 - spatial.distance.cosine(attention_compare_matrix[1], attention_compare_matrix[6]))
    fts.append(1 - spatial.distance.cosine(attention_compare_matrix[2], attention_compare_matrix[6]))
    tts.append(1 - spatial.distance.cosine(attention_compare_matrix[3], attention_compare_matrix[6]))
    print(f"cos std: wosemdc {ffs[-1]}")
    print(f"cos std: wosem {tfs[-1]}")
    print(f"cos std: wodc {fts[-1]}")
    print(f"cos std: semdc {tts[-1]}")

    ff.append(1 - spatial.distance.cosine(attention_compare_matrix[0], attention_compare_matrix[5]))
    tf.append(1 - spatial.distance.cosine(attention_compare_matrix[1], attention_compare_matrix[5]))
    ft.append(1 - spatial.distance.cosine(attention_compare_matrix[2], attention_compare_matrix[5]))
    tt.append(1 - spatial.distance.cosine(attention_compare_matrix[3], attention_compare_matrix[5]))
    print(f"cos anomaly: wosemdc {ff[-1]}")
    print(f"cos anomaly: wosem {tf[-1]}")
    print(f"cos anomaly: wodc {ft[-1]}")
    print(f"cos anomaly: semdc {tt[-1]}")

    ff.append(1 - spatial.distance.cosine(attention_compare_matrix[0], attention_compare_matrix[4]))
    tf.append(1 - spatial.distance.cosine(attention_compare_matrix[1], attention_compare_matrix[4]))
    ft.append(1 - spatial.distance.cosine(attention_compare_matrix[2], attention_compare_matrix[4]))
    tt.append(1 - spatial.distance.cosine(attention_compare_matrix[3], attention_compare_matrix[4]))
    print(f"cos valuable: wosemdc {ff[-1]}")
    print(f"cos valuable: wosem {tf[-1]}")
    print(f"cos valuable: wodc {ft[-1]}")
    print(f"cos valuable: semdc {tt[-1]}")

    print(
        f"cos anomaly and std: semdc {1 - spatial.distance.cosine(attention_compare_matrix[5], attention_compare_matrix[6])}")
    print(
        f"cos anomaly and valuable: semdc {1 - spatial.distance.cosine(attention_compare_matrix[5], attention_compare_matrix[4])}")
    print(
        f"cos std and valuable: semdc {1 - spatial.distance.cosine(attention_compare_matrix[6], attention_compare_matrix[4])}")
    print(center)

if __name__ == '__main__':
    main()
