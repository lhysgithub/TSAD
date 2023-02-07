from utils import *
from torch.utils.data import sampler
from torch.utils.data import DataLoader, random_split, Subset, SubsetRandomSampler
from preprocess_stgat import build_loc_net
from net_struct_stgat import get_feature_map, get_fc_graph_struc, get_tc_graph_struc


class TimeSeriesDatabase:
    def __init__(self, args):
        (self.x_train, self.y_train), (x_test, y_test) = get_data_from_source(args, normalize=True)
        self.x_val, self.y_val, self.x_test, self.y_test = split_val_set_v2(x_test, y_test, args.val_split)
        train_set = [self.x_train, self.y_train, self.x_val, self.y_val]
        test_set = [self.x_train, self.y_train, x_test, y_test]
        self.args = args
        self.args.n_features = self.x_train.shape[1]
        self.batchsz = 1
        self.device = args.device
        self.iner_bs = self.args.bs
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": train_set, "test": test_set}
        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"], mode='train'),
                               "test": self.load_data_cache(self.datasets["test"], mode='test')}

    def load_data_cache(self, data_pack, mode='train'):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        data_cache = []
        if mode == 'train':
            # select = np.random.randint(0, 2, self.args.n_features)
            # self.select = select
            # self.target_dims = bool_list2list(select)
            self.edge_index = get_adge_index(self.x_train, self.args, list(range(self.args.n_features)))
            # temp_edge_index = get_adge_index(self.x_train, self.args, self.target_dims)
            train_dataset = TimeDataset(data_pack[0], data_pack[1], self.edge_index, mode='train', config=self.args)
            # train_dataset = TimeDataset(sample_input_by_bool(select, data_pack[0]), data_pack[1], temp_edge_index, mode='train', config=self.args)
            test_dataset = TimeDataset(data_pack[2], data_pack[3], self.edge_index, mode='test', config=self.args)
            spt_loader = DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=0)
            qry_loader = DataLoader(test_dataset, batch_size=self.args.batch, shuffle=False, num_workers=0)
        else:
            train_dataset = TimeDataset(data_pack[0], data_pack[1], self.edge_index,  mode='train', config=self.args)
            test_dataset = TimeDataset(data_pack[2], data_pack[3], self.edge_index,  mode='test', config=self.args)
            spt_loader = DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=0)
            qry_loader = DataLoader(test_dataset, batch_size=self.args.batch, shuffle=False, num_workers=0)
        data_cache.append([spt_loader, qry_loader])
        return data_cache

    def next(self, mode='train'):
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode], mode=mode)

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch


class TimeDataset(Dataset):
    def __init__(self, data, labels,  edge_index, mode='train', config=None):
        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        self.x, self.y, self.x_row, self.labels,self.selecteds = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr = [], []
        x_row_arr = []
        labels_arr = []
        selected_arr = []
        # slide_win = self.config["slide_win"]
        # slide_stride= self.config[k]
        # slide_win, slide_stride = [self.config[k] for k in ['slide_win', 'slide_stride']]
        is_train = self.mode == 'train'
        total_time_len, node_num = data.shape

        # 如果为训练数据集，则返回窗口起始位置到数据集末尾，步长为slide_stride的滑窗索引，如果为其他数据集则返回步长为1的滑窗索引
        rang = range(self.config.slide_win, total_time_len, self.config.slide_stride) if is_train else range(self.config.slide_win, total_time_len)

        for i in rang:
            ft = data[i - self.config.slide_win:i,:]  # 0~14条
            tar = data[i, :]  # 第15条
            select = np.random.randint(0, 2, self.config.n_features)
            if np.random.random() < 0.5:
                select = np.ones_like(select)
            # selected = bool_list2list(select)
            x_row_arr.append(ft)
            if is_train:
                ft = sample_input_by_bool(select, ft)
            x_arr.append(ft)
            y_arr.append(tar)
            labels_arr.append(labels[i])
            selected_arr.append(np.array(select))

        x = np.array(x_arr)
        y = np.array(y_arr)
        x_row = np.array(x_row_arr)
        labels = np.array(labels_arr)
        selecteds = np.array(selected_arr)

        return x, y,x_row, labels, selecteds

    def __getitem__(self, idx):
        row = self.x_row[idx]
        feature = self.x[idx]
        y = self.y[idx]
        fc_edge_index = self.edge_index[0].long()
        tc_edge_index = self.edge_index[1].long()

        label = self.labels[idx]
        selected = self.selecteds[idx]

        return row, feature, y, label, selected, fc_edge_index, tc_edge_index


def get_adge_index(train, config, all_features):
    feature_map = list(range(0, train.shape[1]))
    fc_struc = get_fc_graph_struc(feature_map)  # 获取所有节点与其他节点的连接关系字典

    fc_edge_index = build_loc_net(fc_struc, all_features, feature_map=feature_map)  # 获取所有节点与其子集节点的连接矩阵
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)  # 将连接矩阵转换成Tensor,torch.Size([2, 702])

    temporal_map = list(range(0, config.slide_win))
    tc_struc = get_tc_graph_struc(config.slide_win)

    tc_edge_index = build_loc_net(tc_struc, temporal_map, feature_map=temporal_map)  # 获取所有节点与其子集节点的连接矩阵
    tc_edge_index = torch.tensor(tc_edge_index, dtype=torch.long)  # 将连接矩阵转换成Tensor,torch.Size([2, 702])

    return (fc_edge_index, tc_edge_index)