#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date : 
'''
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
from preprocess_stgat import build_loc_net
from net_struct_stgat import get_feature_map, get_fc_graph_struc, get_tc_graph_struc
from utils import *


class TimeDataset(Dataset):
    def __init__(self, data, labels,  edge_index, mode='train', config=None):
        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []
        # slide_win, slide_stride, pre_term = [self.config[k] for k in ['slide_win', 'slide_stride', 'pre_term']]
        slide_win, slide_stride, pre_term = self.config.slide_win, self.config.slide_stride, self.config.pre_term
        is_train = self.mode == 'train'
        total_time_len, node_num = data.shape

        # 如果为训练数据集，则返回窗口起始位置到数据集末尾，步长为slide_stride的滑窗索引，如果为其他数据集则返回步长为1的滑窗索引
        # rang = range(slide_win, total_time_len-pre_term+1, slide_stride) if is_train else range(slide_win, total_time_len-pre_term+1)
        # rang = range(slide_win, total_time_len - slide_stride) if is_train else range(slide_win, total_time_len - slide_stride, slide_stride)
        # rang = range(slide_win, total_time_len - slide_stride) if is_train else range(slide_win,
        #                                                                               total_time_len - slide_stride)
        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win,
                                                                                     total_time_len, slide_stride)
        # if self.config.train_stride:
        # rang = range(slide_win, total_time_len - slide_stride,slide_stride) if is_train else range(slide_win,total_time_len - slide_stride,slide_stride)
        if self.config.only_variation_time:
            # rang = [slide_win,slide_win*13,slide_win*18,slide_win*23] # 1/2
            rang = [slide_win, slide_win*5, slide_win*13,
                    slide_win*15, slide_win*18, slide_win*23]  # 1/3
            # rang = [slide_win,slide_win*3,slide_win*5,slide_win*13,slide_win*15,slide_win*18,slide_win*20,slide_win*23] # 1/4
            # rang = [slide_win,slide_win*3,slide_win*5,slide_win*7,slide_win*13,slide_win*15,slide_win*18,slide_win*20,slide_win*21,slide_win*23] # 1/5
            # rang = range(slide_win, total_time_len,slide_stride) if is_train else range(slide_win,total_time_len,slide_stride) 1/6

            # slide_stride = 288
            # rang = range(13*12-slide_win, total_time_len - slide_stride,slide_stride) if is_train else range(13*12-slide_win,
            #                                                                           total_time_len - slide_stride,slide_stride)
        # rang = range(slide_win, total_time_len - slide_stride) if is_train else range(slide_win,
            #   total_time_len - slide_stride,slide_stride)
        for i in rang:
            ft = data[i - slide_win:i, :]  # 0~14条
            tar = data[i+pre_term-1, :]  # 第15条
            if self.config.condition_control:
                # add future limited todo de-comment for predict future with control
                tar_limited = np.expand_dims(tar, axis=0)
                if self.config.condition_infer and not is_train:
                    range_ = self.config.scaler.data_range_[-1]
                    max_ = self.config.scaler.data_max_[-1]
                    min_ = self.config.scaler.data_min_[-1]
                    norm_tar_limited = (
                        0.01*self.config.condition_variate - min_)/range_
                    tar_limited = np.ones_like(
                        tar_limited)*norm_tar_limited
                tar_limited_repeat = np.repeat(
                    tar_limited, len(ft), axis=0)[:, -1]
                ft = np.concatenate(
                    [ft, np.expand_dims(tar_limited_repeat, axis=1)], axis=1)
                tt = ft.T
                # tar = np.concatenate([tar, np.array([0.1])], axis=0)# done notice 这里直接使用了限制比例，作为这一位维度的值
            #
            x_arr.append(ft)
            y_arr.append(tar)
            labels_arr.append(labels[i+pre_term-1])

        x = np.array(x_arr)
        y = np.array(y_arr)
        labels = np.array(labels_arr)

        return x, y, labels

    def __getitem__(self, idx):
        feature = self.x[idx]
        y = self.y[idx]
        fc_edge_index = self.edge_index[0].long()
        tc_edge_index = self.edge_index[1].long()

        label = self.labels[idx]

        return feature, y, label, fc_edge_index, tc_edge_index


def get_adge_index(train, config):
    feature_map = list(range(0, train.shape[1]))
    if config.condition_control:
        feature_map = list(range(0, train.shape[1]+1))
    fc_struc = get_fc_graph_struc(feature_map)  # 获取所有节点与其他节点的连接关系字典

    fc_edge_index = build_loc_net(
        fc_struc, feature_map, feature_map=feature_map)  # 获取所有节点与其子集节点的连接矩阵
    # 将连接矩阵转换成Tensor,torch.Size([2, 702])
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

    temporal_map = list(range(0, config.slide_win))
    tc_struc = get_tc_graph_struc(config.slide_win)

    tc_edge_index = build_loc_net(
        tc_struc, temporal_map, feature_map=temporal_map)  # 获取所有节点与其子集节点的连接矩阵
    # 将连接矩阵转换成Tensor,torch.Size([2, 702])
    tc_edge_index = torch.tensor(tc_edge_index, dtype=torch.long)

    return (fc_edge_index, tc_edge_index)


def data_load(config):
    (train, train_label), (test, test_label) = get_data_from_source(config)

    edge_index = get_adge_index(train, config)

    cfg = {
        'slide_win': config.slide_win,
        'slide_stride': config.slide_stride,
        'pre_term': config.pre_term,
    }

    train_dataset = TimeDataset(
        train, train_label, edge_index, mode='train', config=config)
    test_dataset = TimeDataset(
        test, test_label, edge_index, mode='test', config=config)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch, shuffle=False, num_workers=0)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch, shuffle=True, num_workers=0)

    return train_dataloader, test_dataloader


def data_load_from_exist_np(config):
    (train, train_label), (test, test_label) = (config.train,
                                                config.train_label), (config.test, config.test_label)

    edge_index = get_adge_index(train, config)

    cfg = {
        'slide_win': config.slide_win,
        'slide_stride': config.slide_stride,
        'pre_term': config.pre_term,
        'condition_control': config.condition_control
    }

    train_dataset = TimeDataset(
        train, train_label, edge_index, mode='train', config=config)
    test_dataset = TimeDataset(
        test, test_label, edge_index, mode='test', config=config)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch, shuffle=False, num_workers=0)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch, shuffle=True, num_workers=0)

    return train_dataloader, test_dataloader
