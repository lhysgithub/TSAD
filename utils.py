import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import json
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler


def get_data_dim(dataset):
    """
    :param dataset: Name of dataset
    :return: Number of dimensions in data
    """
    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 55
    elif str(dataset).startswith("machine"):
        return 38
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_target_dims(dataset):
    """
    :param dataset: Name of dataset
    :return: index of data dimension that should be modeled (forecasted and reconstructed),
                     returns None if all input dimensions should be modeled
    """
    if dataset == "SMAP":
        return [0]  # None # [0]
    elif dataset == "MSL":
        return [0]  # None # [0]
    elif dataset == "ZX":
        return [0]  # None # [10] [0]
    else:
        return None
    # else:
    #     raise ValueError("unknown dataset " + str(dataset))


def get_dataset_np(config):
    dataset = config.dataset
    if "WT" in dataset:
        variable = config.group
        train_df = pd.read_csv(f'./data/WT/{variable}/train_orig.csv', sep=",", header=None, dtype=np.float32).dropna(
            axis=0)
        test_df = pd.read_csv(f'./data/WT/{variable}/test_orig.csv', sep=",", header=None, dtype=np.float32).dropna(
            axis=0)
        dim = 10
        # train_df["y"] = np.zeros(train_df.shape[0], dtype=np.float32)

        # Get test anomaly labels
        test_label = test_df.iloc[:, dim]
        test_df.drop(dim, axis=1, inplace=True)
        return train_df.to_numpy(), test_df.to_numpy(), test_label.to_numpy()
    elif "SMD" in dataset:
        variable = config.group
        train_df = pd.read_csv(f'./data/SMD/train/machine-{variable}.txt', header=None, sep=",", dtype=np.float32)
        test_df = pd.read_csv(f'./data/SMD/test/machine-{variable}.txt', header=None, sep=",", dtype=np.float32)

        # Get test anomaly labels
        test_labels = np.genfromtxt(f'./data/SMD/test_label/machine-{variable}.txt', dtype=np.float32, delimiter=',')
        return train_df.to_numpy(), test_df.to_numpy(), test_labels
    elif "SMAP" in dataset:
        variable = config.group
        train = np.load(f'./data/SMAP/train/{variable}.npy')
        test = np.load(f'./data/SMAP/test/{variable}.npy')
        test_label = np.zeros(len(test), dtype=np.float32)

        # Set test anomaly labels from files
        labels = pd.read_csv(f'./data/SMAP/labeled_anomalies.csv', sep=",", index_col="chan_id")
        label_str = labels.loc[variable, "anomaly_sequences"]
        label_list = json.loads(label_str)
        for i in label_list:
            test_label[i[0]:i[1] + 1] = 1.0
        return train, test, test_label
    elif "MSL" in dataset:
        variable = config.group
        train = np.load(f'./data/SMAP/train/{variable}.npy')
        test = np.load(f'./data/SMAP/test/{variable}.npy')
        test_label = np.zeros(len(test), dtype=np.float32)

        # Set test anomaly labels from files
        labels = pd.read_csv(f'./data/SMAP/labeled_anomalies.csv', sep=",", index_col="chan_id")
        label_str = labels.loc[variable, "anomaly_sequences"]
        label_list = json.loads(label_str)
        for i in label_list:
            test_label[i[0]:i[1] + 1] = 1.0
        return train, test, test_label
    elif "WADI" in dataset:
        variable = config.group
        if variable == "A2":
            dim = 82
        elif variable == "A1":
            dim = 127
        train_df = pd.read_csv(f'./data/WADI/{variable}/train.csv', sep=",", header=None, skiprows=1,
                               dtype=np.float32).fillna(0)
        test_df = pd.read_csv(f'./data/WADI/{variable}/test.csv', sep=",", header=None, skiprows=1,
                              dtype=np.float32).fillna(0)
        # train_df["y"] = np.zeros(train_df.shape[0], dtype=np.float32)

        # Get test anomaly labels
        test_label = test_df.iloc[:, dim]
        test_df.drop(dim, axis=1, inplace=True)
        return train_df.to_numpy(), test_df.to_numpy(), test_label.to_numpy()
    elif "SWAT" in dataset:
        variable = config.group
        dim = 0
        if variable == "A1_A2":
            dim = 50
        elif variable == "A4_A5":
            dim = 77
        train_df = pd.read_csv(f'./data/SWAT/{variable}/train.csv', sep=",", header=None, skiprows=1,
                               dtype=np.float32).fillna(0)
        test_df = pd.read_csv(f'./data/SWAT/{variable}/test.csv', sep=",", header=None, skiprows=1,
                              dtype=np.float32).fillna(0)
        # train_df["y"] = np.zeros(train_df.shape[0], dtype=np.float32)

        # Get test anomaly labels
        test_label = test_df.iloc[:, dim]
        train_df.drop(dim, axis=1, inplace=True)
        test_df.drop(dim, axis=1, inplace=True)
        return train_df.to_numpy(), test_df.to_numpy(), test_label.to_numpy()
    elif "BATADAL" in dataset:
        variable = config.group
        dim = 43
        train_df = pd.read_csv(f'./data/BATADAL/train.csv', sep=",", header=None, skiprows=1,
                               dtype=np.float32).fillna(0)
        test_df = pd.read_csv(f'./data/BATADAL/test.csv', sep=",", header=None, skiprows=1,
                              dtype=np.float32).fillna(0)
        # train_df["y"] = np.zeros(train_df.shape[0], dtype=np.float32)

        # Get test anomaly labels
        test_label = test_df.iloc[:, dim]
        train_df.drop(dim, axis=1, inplace=True)
        test_df.drop(dim, axis=1, inplace=True)
        return train_df.to_numpy(), test_df.to_numpy(), test_label.to_numpy()
    elif "ZX" in dataset:
        variable = config.group
        # dim = 13
        train_df = pd.read_csv(f'./data/ZX/train/{variable}.csv', sep=",", header=None, skiprows=1).fillna(0)
        test_df = pd.read_csv(f'./data/ZX/test/{variable}.csv', sep=",", header=None, skiprows=1).fillna(0)
        train_df.drop(0, axis=1, inplace=True)
        test_df.drop(0, axis=1, inplace=True)
        dim = len(list(train_df)) - 1
        # train_df["y"] = np.zeros(train_df.shape[0], dtype=np.float32)

        # Get test anomaly labels
        test_label = test_df.iloc[:, dim]
        train_df.drop(dim, axis=1, inplace=True)
        test_df.drop(dim, axis=1, inplace=True)
        return train_df.to_numpy(), test_df.to_numpy(), test_label.to_numpy()


def get_dim(args):
    n_features = 0
    if args.dataset == "SMD":
        n_features = 38
    elif args.dataset == "SMAP":
        n_features = 25
    elif args.dataset == "MSL":
        n_features = 55
    elif args.dataset == "WT":
        n_features = 10
    elif args.dataset == "WADI":
        if args.group == "A1":
            n_features = 127
        else:
            n_features = 82
    elif args.dataset == "SWAT":
        if args.group == "A1_A2":
            n_features = 50
        else:
            n_features = 77
    elif args.dataset == "BATADAL":
        n_features = 43
    elif args.dataset == "ZX":
        n_features = 13
        if args.condition_control:
            n_features = 14
    return n_features


def get_data_from_source(args, normalize=True):
    train_data, test_data, test_label = get_dataset_np(args)
    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None)
        test_data, _ = normalize_data(test_data, scaler=scaler)
        args.scaler = scaler

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", None if test_label is None else test_label.shape)
    train_label = np.zeros(len(train_data), dtype=np.float32)
    return (train_data, train_label), (test_data, test_label)


# def get_data(dataset, max_train_size=None, max_test_size=None,normalize=False, spec_res=False, train_start=0, test_start=0):
#     """
#     Get data from pkl files
#
#     return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
#     Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
#     """
#     prefix = "datasets"
#     if str(dataset).startswith("machine"):
#         prefix += "/ServerMachineDataset/processed"
#     elif dataset in ["MSL", "SMAP"]:
#         prefix += "/data/processed"
#     if max_train_size is None:
#         train_end = None
#     else:
#         train_end = train_start + max_train_size
#     if max_test_size is None:
#         test_end = None
#     else:
#         test_end = test_start + max_test_size
#     print("load data of:", dataset)
#     print("train: ", train_start, train_end)
#     print("test: ", test_start, test_end)
#     x_dim = get_data_dim(dataset)
#     f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
#     train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
#     f.close()
#     try:
#         f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
#         test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
#         f.close()
#     except (KeyError, FileNotFoundError):
#         test_data = None
#     try:
#         f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
#         test_label = pickle.load(f).reshape((-1))[test_start:test_end]
#         f.close()
#     except (KeyError, FileNotFoundError):
#         test_label = None
#
#     if normalize:
#         train_data, scaler = normalize_data(train_data, scaler=None)
#         test_data, _ = normalize_data(test_data, scaler=scaler)
#
#     print("train set shape: ", train_data.shape)
#     print("test set shape: ", test_data.shape)
#     print("test set label shape: ", None if test_label is None else test_label.shape)
#     return (train_data, None), (test_data, test_label)


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        # index = index * self.window
        x = self.data[index: index + self.window]
        y = self.data[index + self.window: index + self.window + self.horizon]
        return x, y

    def __len__(self):
        # return int(len(self.data) / self.window) - 1
        # return int(len(self.data) - self.window) - 1
        return int(len(self.data) - self.window)


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.plot(losses["train_recon"], label="Recon loss")
    plt.plot(losses["train_total"], label="Total loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.plot(losses["val_recon"], label="Recon loss")
    plt.plot(losses["val_total"], label="Total loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()


def load(model, PATH, device="cpu"):
    """
    Loads the model's parameters from the path mentioned
    :param PATH: Should contain pickle file
    """
    model.load_state_dict(torch.load(PATH, map_location=device))


def get_series_color(y):
    if np.average(y) >= 0.95:
        return "black"
    elif np.average(y) == 0.0:
        return "black"
    else:
        return "black"


def get_y_height(y):
    if np.average(y) >= 0.95:
        return 1.5
    elif np.average(y) == 0.0:
        return 0.1
    else:
        return max(y) + 0.1


def adjust_anomaly_scores(scores, dataset, is_train, lookback):
    """
    Method for MSL and SMAP where channels have been concatenated as part of the preprocessing
    :param scores: anomaly_scores
    :param dataset: name of dataset
    :param is_train: if scores is from train set
    :param lookback: lookback (window size) used in model
    """

    return scores

    # ？没懂
    # Remove errors for time steps when transition to new channel (as this will be impossible for model to predict)
    if dataset.upper() not in ['SMAP', 'MSL']:
        return scores

    adjusted_scores = scores.copy()
    if is_train:
        md = pd.read_csv(f'data/SMAP/{dataset.lower()}_train_md.csv')
    else:
        md = pd.read_csv('data/SMAP/labeled_anomalies.csv')
        md = md[md['spacecraft'] == dataset.upper()]

    md = md[md['chan_id'] != 'P-2']

    # Sort values by channel
    md = md.sort_values(by=['chan_id'])

    # Getting the cumulative start index for each channel
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    sep_cuma = sep_cuma[:-1]
    buffer = np.arange(1, 20)
    i_remov = np.sort(np.concatenate((sep_cuma, np.array([i + buffer for i in sep_cuma]).flatten(),
                                      np.array([i - buffer for i in sep_cuma]).flatten())))
    i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
    i_remov = np.sort(np.unique(i_remov))
    if len(i_remov) != 0:
        adjusted_scores[i_remov] = 0

    # Normalize each concatenated part individually
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    s = [0] + sep_cuma.tolist()
    for c_start, c_end in [(s[i], s[i + 1]) for i in range(len(s) - 1)]:
        e_s = adjusted_scores[c_start: c_end + 1]

        e_s = (e_s - np.min(e_s)) / (np.max(e_s) - np.min(e_s))
        adjusted_scores[c_start: c_end + 1] = e_s

    return adjusted_scores


def get_f1(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["f1"]
    return f1


def get_key_from_bf_result(file_name, key):
    try:
        with open(file_name) as f:
            summary = json.load(f)
            f1 = summary["bf_result"][key]
        return f1
    except Exception as e:
        print(e)
        return 0.0


def get_key_for_maml(file_name, key):
    try:
        with open(file_name) as f:
            summary = json.load(f)
            f1 = summary[key]
        return f1
    except Exception as e:
        print(e)
        return 0.0


def get_f1_for_maml(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["f1"]
    return f1


def get_f1_for_omni(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["best-f1"]
    return f1


def get_key_for_omni(file_name, key):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary[key]
    return f1


def get_f1_for_omni_broken(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            if "f1" in line:
                f1 = float(line.split(":")[1].split(",")[0])
    return f1


def get_key_from_omni_broken(file_name, key):
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            if key in line:
                f1 = float(line.split(":")[1].split(",")[0])
    return f1


def get_precision(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["precision"]
    return f1


def get_recall(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["recall"]
    return f1


def list2bin(l, max_dim):
    bin = []
    for i in range(max_dim):
        if i in l:
            bin.append("1")
        else:
            bin.append("0")
    return "".join(bin)


def bool_list2bin(l):
    bin = []
    for i in l:
        if i:
            bin.append("1")
        else:
            bin.append("0")
    return "".join(bin)


def bin2list(bin):
    l = []
    for i in range(len(bin)):
        if bin[i] == "1":
            l.append(i)
    return l


def bool_list2list(l):
    l2 = []
    for i in range(len(l)):
        if l[i]:
            l2.append(i)
    return l2


def filter_input(select, x_train):
    temp_x_train = []
    for i in range(len(select)):
        xi = x_train[:, select[i]].reshape(-1, 1)
        if i == 0:
            temp_x_train = xi
        else:
            temp_x_train = np.concatenate((temp_x_train, xi), axis=1)
    return temp_x_train


def filter_train_test_set(args, select, x_train, x_test):
    temp_x_train = filter_input(select, x_train)
    temp_x_test = filter_input(select, x_test)
    if args.normalize:
        temp_x_train, scaler = normalize_data(temp_x_train, scaler=None)
        temp_x_test, _ = normalize_data(temp_x_test, scaler=scaler)
    return temp_x_train, temp_x_test


def filter_input_by_bool(select, x_train):
    temp_x_train = []
    first = 0
    for i in range(len(select)):
        xi = x_train[:, i].reshape(-1, 1)
        if select[i]:
            if first == 0:
                temp_x_train = xi
                first = 1
            else:
                temp_x_train = np.concatenate((temp_x_train, xi), axis=1)
    return temp_x_train


def sample_input_by_bool(select, x_train):
    temp_x_train = []
    first = 0
    for i in range(len(select)):
        xi = x_train[:, i].reshape(-1, 1)
        if i == 0:
            temp_x_train = xi
            first = 1
            continue
        if select[i]:
            if first == 0:
                temp_x_train = xi
                first = 1
            else:
                temp_x_train = np.concatenate((temp_x_train, xi), axis=1)
        else:
            if first == 0:
                temp_x_train = np.zeros_like(xi)
                first = 1
            else:
                temp_x_train = np.concatenate((temp_x_train, np.zeros_like(xi)), axis=1)
    return temp_x_train


def split_val_set(x_test, y_test, val_ratio=0.05):
    dataset_len = int(len(x_test))
    val_use_len = int(dataset_len * val_ratio)
    index_list = []
    lens_list = []
    find = 0
    count = 0
    for i in range(len(y_test)):
        if int(y_test[i]) == 1:
            if find == 0:
                index_list.append(i)
            find = 1
            count += 1
        elif find == 1:
            find = 0
            lens_list.append(count)
            count = 0
    index = random.choice(index_list)
    # index = 0
    # i = np.argmax(lens_list)
    # index = index_list[i]
    start = 0
    end = 0
    if index < val_use_len / 2:
        start = 0
        end = val_use_len
    elif dataset_len - index < val_use_len / 2:
        start = dataset_len - val_use_len
        end = dataset_len
    else:
        start = index - val_use_len / 2
        end = index + val_use_len / 2
    start = int(start)
    end = int(end)
    x_val = x_test[start:end]
    y_val = y_test[start:end]
    new_x_test = np.concatenate((x_test[:start], x_test[end:]))
    new_y_test = np.concatenate((y_test[:start], y_test[end:]))
    return x_val, y_val, new_x_test, new_y_test


def split_val_set_v2(x_test, y_test, val_ratio=0.05):
    dataset_len = int(len(x_test))
    val_use_len = int(dataset_len * val_ratio)
    index_list = []
    lens_list = []
    find = 0
    count = 0
    for i in range(len(y_test)):
        if int(y_test[i]) == 1:
            if find == 0:
                index_list.append(i)
            find = 1
            count += 1
        elif find == 1:
            find = 0
            lens_list.append(count)
            count = 0
    index = random.choice(index_list)
    # index = 0
    # i = np.argmax(lens_list)
    # index = index_list[i]
    starts, ends = get_starts_ends(index_list, val_use_len, dataset_len)
    x_val = []
    y_val = []
    new_x_test = []
    new_y_test = []
    starts = [0] + starts
    ends = [0] + ends
    for i in range(1, len(starts)):
        x_val.append(x_test[starts[i]:ends[i]])
        y_val.append(y_test[starts[i]:ends[i]])
        new_x_test.append(x_test[ends[i - 1]:starts[i]])
        new_y_test.append(y_test[ends[i - 1]:starts[i]])
    new_x_test.append(x_test[ends[len(starts) - 1]:])
    new_y_test.append(y_test[ends[len(starts) - 1]:])
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)
    new_x_test = np.concatenate(new_x_test)
    new_y_test = np.concatenate(new_y_test)
    return x_val, y_val, new_x_test, new_y_test


def get_starts_ends(indexes, val_use_len, dataset_len):
    anomaly_types = len(indexes)
    use_len_per_type = int(val_use_len / anomaly_types)
    starts = []
    ends = []
    for i in range(anomaly_types):
        # multi examples 需要小心使用，小心选出来的val直接包含了全部的abnormal examples。
        # 并且，如果包含全部abnormal examples的val仍然不能指导测试集区分开normal和abnormal，则可能是异常难以区分的问题，而非模型设计的不好。
        s = indexes[i] - int(use_len_per_type/2)
        e = indexes[i] + int(use_len_per_type/2)
        #
        # few examples
        # if i == 0:
        #     s = indexes[i] - 100
        #     e = indexes[i] + 1
        # else:
        #     s = indexes[i] - 99
        #     e = indexes[i] + 1
        #
        starts.append(s if s >= 0 else 0)
        ends.append(e if e < dataset_len else dataset_len)
    return starts, ends


def print_info(epoch, train_mean_loss, test_mean_loss, test_recon_pre_loss, bf_eval, iter_time):
    print(
        f'[Epoch {epoch + 1:.2f}] | Train loss: {train_mean_loss:.4f} | '
        f'Test Loss: {test_mean_loss:.2f} | Test_recon_pre_loss: {test_recon_pre_loss:.4f} | '
        f'best F1: {bf_eval["f1"]:.4f} | precision: '
        f'{bf_eval["precision"]:.4f} | Recall: {bf_eval["recall"]:.4f} | Time: {iter_time:.4f}')
    return {
        'epoch': epoch + 1,
        'loss': test_mean_loss,
        'precision': bf_eval["precision"],
        'recall': bf_eval["recall"],
        'f1': bf_eval["f1"],
        'TP': int(bf_eval["TP"]),
        'TN': int(bf_eval["TN"]),
        'FP': int(bf_eval["FP"]),
        'FN': int(bf_eval["FN"]),
    }


def plot_double_curve(data1, data2, name1, name2, filename):
    plt.cla()
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(data1)), data1, color='blue', label=name1)
    plt.plot(range(len(data2)), data2, color='red', label=name2, alpha=0.5)
    plt.legend()
    plt.savefig(filename)


def compute_std_mse(data1, data2):
    d1, scaler = normalize_data(data1, scaler=None)
    d2, _ = normalize_data(data2, scaler=scaler)
    predicts_loss = np.sqrt((np.array(d2) - d1) ** 2).mean()
    return predicts_loss


def get_std_data(data1, data2):
    d1, scaler = normalize_data(data1, scaler=None)
    d2, _ = normalize_data(data2, scaler=scaler)
    return d1,d2


def up_down_mse(data1, data2, scope_length=12):
    data1_max = []
    data1_min = []
    data2_max = []
    data2_min = []
    for i in range(0, len(data1), scope_length):
        data1_temp_set = data1[i:i + scope_length]
        data2_temp_set = data2[i:i + scope_length]
        data1_max.append(np.max(data1_temp_set, axis=1))
        data1_min.append(np.min(data1_temp_set, axis=1))
        data2_max.append(np.max(data2_temp_set, axis=1))
        data2_min.append(np.min(data2_temp_set, axis=1))
    up_loss = np.sqrt((np.array(data1_max) - np.array(data2_max)) ** 2).mean(axis=1)
    down_loss = np.sqrt((np.array(data1_min) - np.array(data2_min)) ** 2).mean(axis=1)
    loss = up_loss + down_loss
    return loss


def up_down_mse_for_hour(data1, data2, scope_length=12):
    data1_max = np.max(data1, axis=1)
    data1_min = np.min(data1, axis=1)
    data2_max = np.max(data2, axis=1)
    data2_min = np.min(data2, axis=1)
    up_loss = np.sqrt((np.array(data1_max) - np.array(data2_max)) ** 2).mean()
    down_loss = np.sqrt((np.array(data1_min) - np.array(data2_min)) ** 2).mean()
    loss = up_loss + down_loss
    return loss


def get_anormal_dimensions(args):
    anormal = []
    file_name = f"data/SMD/interpretation_label/machine-{args.group}.txt"
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            temp_list = line.split(":")[1].split(",")
            anormal.append([int(i) for i in temp_list])
    return anormal


def normalization_axis0(data):
    _range = np.max(data,axis=0) - np.min(data,axis=0)
    return (data - np.min(data,axis=0)) / _range


def normalization_axis0_st(data):
    return (data - np.mean(data))/np.std(data)


def get_attention_from_model(loader, device, model):
    device = "cuda" if device and torch.cuda.is_available() else "cpu"
    attentions = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            a = model.get_gat_attention(x)
            attentions.append(a.detach().cpu().numpy())
    attentions = np.concatenate(attentions, axis=0)
    return attentions