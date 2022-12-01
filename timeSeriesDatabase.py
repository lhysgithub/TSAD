from utils import *


class TimeSeriesDatabase:
    def __init__(self, args, device=None):
        (self.x_train, self.y_train), (x_test, y_test) = get_data_from_source(args, normalize=True)
        self.x_val, self.y_val, self.x_test, self.y_test = split_val_set(x_test, y_test, args.val_split)
        train_set = [self.x_train, self.y_train, self.x_val, self.y_val]
        test_set = [self.x_train, self.y_train, self.x_test, self.y_test]
        self.args = args
        self.args.n_features = self.x_train.shape[1]
        self.args.bs = 1
        # self.args.lookback = 1000
        self.batchsz = self.args.bs

        self.device = args.device
        # self.k_shot = int(len(self.x_train) / self.args.lookback)
        # self.k_query = int(len(self.x_test) / self.args.lookback)
        # self.k_shot = 20
        # self.k_query = 20
        self.iner_bs = 256
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
        x_spts, y_spts, z_spts, x_qrys, y_qrys, z_qrys = [], [], [], [], [], []
        if mode == "train":
            select = np.random.randint(0, 2, self.args.n_features)
            x_spt, z_spt = self.get_samples_test_np(sample_input_by_bool(select, data_pack[0]))
            x_spts.append(x_spt)
            z_spts.append(z_spt)
            y_spts.append(self.get_samples_test_np(data_pack[1])[0])
            x_qry, z_qry = self.get_samples_test_np(sample_input_by_bool(select, data_pack[2]))
            x_qrys.append(x_qry)
            z_qrys.append(z_qry)
            y_qrys.append(self.get_samples_test_np(data_pack[3])[0])

        else:
            x_spt, z_spt = self.get_samples_test_np(data_pack[0])
            x_spts.append(x_spt)
            z_spts.append(z_spt)
            y_spts.append(self.get_samples_test_np(data_pack[1])[0])
            x_qry, z_qry = self.get_samples_test_np(data_pack[2])
            x_qrys.append(x_qry)
            z_qrys.append(z_qry)
            y_qrys.append(self.get_samples_test_np(data_pack[3])[0])
        # x_spts, y_spts, z_spts, x_qrys, y_qrys, z_qrys = [torch.from_numpy(np.array(z)).to(self.device) for z in
        #                                                   [x_spts, y_spts, z_spts, x_qrys, y_qrys, z_qrys]]
        x_spts, y_spts, z_spts, x_qrys, y_qrys, z_qrys = [np.array(z) for z in
                                                          [x_spts, y_spts, z_spts, x_qrys, y_qrys, z_qrys]]
        data_cache.append([x_spts, y_spts, z_spts, x_qrys, y_qrys, z_qrys])
        return data_cache

    def get_samples(self, data_pack):
        dataset = SlidingWindowDataset(data_pack, self.args.lookback)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
        temp = []
        for x, y in dataloader:
            temp.append(x)
        return torch.stack(temp)

    def get_samples_np(self, data_pack):
        dataset = SlidingWindowDataset(data_pack, self.args.lookback)
        temp = []
        # length = int(len(data_pack)/self.args.lookback)
        length = self.k_shot
        for i in range(length):
            temp.append(dataset[i][0])
        return np.array(temp)

    def get_samples_test_np(self, data_pack):
        dataset = SlidingWindowDataset(data_pack, self.args.lookback)
        x = []
        z = []
        xs = []
        zs = []
        # length = int(len(data_pack)/self.args.lookback)
        length = len(dataset)
        # length = self.k_shot
        count = 0
        for i in range(length):
            if count < self.iner_bs:
                x.append(dataset[i][0])
                z.append(dataset[i][1])
                count += 1
            else:
                xs.append(np.array(x))
                zs.append(np.array(z))
                count = 1
                x = [dataset[i][0]]
                z = [dataset[i][1]]
        return np.array(xs), np.array(zs)

    def next(self, mode='train'):
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode], mode='train')

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch


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
        # return int(len(self.data) / self.window)
        return int(len(self.data) - self.window)