from utils import *
from torch.utils.data import sampler
from torch.utils.data import DataLoader, random_split, Subset, SubsetRandomSampler


class TimeSeriesDatabase:
    def __init__(self, args, device=None):
        (self.x_train, self.y_train), (x_test, y_test) = get_data_from_source(args, normalize=True)
        self.x_val, self.y_val, self.x_test, self.y_test = split_val_set_v2(x_test, y_test, args.val_split)
        train_set = [self.x_train, self.y_train, self.x_val, self.y_val]
        test_set = [self.x_train, self.y_train, x_test, y_test]
        self.args = args
        self.args.n_features = self.x_train.shape[1]
        # self.args.bs = 1
        # self.args.lookback = 1000
        self.batchsz = 1

        self.device = args.device
        # self.k_shot = int(len(self.x_train) / self.args.lookback)
        # self.k_query = int(len(self.x_test) / self.args.lookback)
        # self.k_shot = 20
        # self.k_query = 20
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
            spt_dataset = SlidingWindowDataset(data_pack[0], data_pack[1], "train", self.args.lookback, self.args,
                                               self.args.target_dims)
            spt_loader = DataLoader(spt_dataset, batch_size=self.iner_bs, shuffle=True)
            # qry_dataset = SlidingWindowDatasetv2(data_pack[2], data_pack[3], "train", self.args.lookback, self.args,
            #                                    self.args.target_dims)
            qry_dataset = SlidingWindowDatasetv2(data_pack[2], data_pack[3], "test", self.args.lookback, self.args,
                                                 self.args.target_dims)
            # qry_loader = DataLoader(qry_dataset, batch_size=1, shuffle=False)
            qry_loader = DataLoader(qry_dataset, batch_size=self.iner_bs, shuffle=False)
        else:
            spt_dataset = SlidingWindowDataset(data_pack[0], data_pack[1], "test", self.args.lookback, self.args,
                                               self.args.target_dims)
            spt_loader = DataLoader(spt_dataset, batch_size=self.iner_bs, shuffle=True)
            qry_dataset = SlidingWindowDataset(data_pack[2], data_pack[3], "test", self.args.lookback, self.args,
                                               self.args.target_dims)
            qry_loader = DataLoader(qry_dataset, batch_size=self.iner_bs, shuffle=False)
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


class SlidingWindowDataset(Dataset):
    def __init__(self, data, label, mode, window, args, target_dim=None, horizon=1):
        self.data = data
        self.label = label
        self.window = window
        self.mode = mode
        self.step = int(self.window / 2)
        self.target_dim = target_dim
        self.horizon = horizon
        self.args = args

    def __getitem__(self, index):
        # index = index * self.step
        # index = index * self.window
        select = np.random.randint(0, 2, self.args.n_features)
        if np.random.random() < 0.5:
            select = np.ones_like(select)
        x = self.data[index: index + self.window]
        row = x
        if self.mode == "train":
            x = sample_input_by_bool(select, x)
        y = self.data[index + self.window: index + self.window + self.horizon]
        z = self.label[index + self.window: index + self.window + self.horizon]
        return row, x, y, z, select

    def __len__(self):
        # return int(len(self.data) / self.step)-1
        # return int(len(self.data) / self.window)
        return int(len(self.data) - self.window)

class SlidingWindowDatasetv2(Dataset):
    def __init__(self, data, label, mode, window, args, target_dim=None, horizon=1):
        self.data = data
        self.label = label
        self.window = window
        self.mode = mode
        self.step = int(self.window / 2)
        self.target_dim = target_dim
        self.horizon = horizon
        self.args = args

    def __getitem__(self, index):
        # index = index * self.step
        index = index * self.window
        select = np.random.randint(0, 2, self.args.n_features)
        if np.random.random() < 0.5:
            select = np.ones_like(select)
        x = self.data[index: index + self.window]
        row = x
        if self.mode == "train":
            x = sample_input_by_bool(select, x)
        y = self.data[index + self.window: index + self.window + self.horizon]
        z = self.label[index + self.window: index + self.window + self.horizon]
        return row, x, y, z, select

    def __len__(self):
        # return int(len(self.data) / self.step)-1
        return int(len(self.data) / self.window)
        # return int(len(self.data) - self.window)