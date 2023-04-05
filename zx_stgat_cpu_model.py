#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date :
'''
import argparse
import numpy as np
import random
import torch
from stgat import STGAT
from build_data import *
from torch import nn
from tqdm import tqdm
from utils import *
from args import get_parser
from eval_methods import bf_search
from sklearn import metrics
import json


def train(config, model, train_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    forecast_criterion = nn.MSELoss().to(config.device)
    recon_criterion = nn.MSELoss().to(config.device)
    losses = []
    model.train()
    # think: check fc tc edge 是否有问题？ lhy 没问题
    for x, y, attack_labels, fc_edge_index, tc_edge_index in train_dataloader:
        x, y, fc_edge_index, tc_edge_index = [item.float().to(config.device) for item in
                                              [x, y, fc_edge_index, tc_edge_index]]
        y = y.unsqueeze(1)
        # 正向传播
        recons, forest = model(x, fc_edge_index, tc_edge_index)
        if config.target_dims is not None:
            x = x[:, :, config.target_dims]  # 将输入数据处理成与重建输出相同的形状,如果为NASA数据集则只取第一个维度的数据
            y = y[:, :, config.target_dims]  # 将输入数据处理成与预测输出相同的形状,如果未NASA数据集则只取第一个维度的数据
        if y.ndim == 3:
            y = y.squeeze(1)

        forest_loss = torch.sqrt(forecast_criterion(y, forest))
        recon_loss = torch.sqrt(
            recon_criterion(x, recons))  # recon_criterion=nn.MSELoss()  + scipy.stats.entropy(x, recons)
        loss = recon_loss + forest_loss

        # 方向梯度下降
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 根据误差函数求导
        optimizer.step()  # 进行一轮梯度下降计算
        losses.append(loss.item())
        # print(f"[epoch {epoch}] train loss: {np.array(losses).mean}")
    train_loss = np.array(losses).mean()
    return train_loss


def test(config, model, test_dataloader):
    recons = []
    predicts = []
    test_label = []
    test_data = []
    model.eval()
    with torch.no_grad():
        for x, y, attack_labels, fc_edge_index, tc_edge_index in test_dataloader:
            x, y, fc_edge_index, tc_edge_index = [item.float().to(config.device) for item in
                                                  [x, y, fc_edge_index, tc_edge_index]]
            y = y.unsqueeze(1)
            if config.condition_control:
                y = torch.cat((y, y[:, :, -1].unsqueeze(1)), dim=2)
            # Shifting input to include the observed value (y) when doing the reconstruction
            recon_x = torch.cat((x[:, 1:, :], y), dim=1)
            window_recon, _ = model(recon_x, fc_edge_index, tc_edge_index)
            _, window_predict = model(x, fc_edge_index, tc_edge_index)

            if config.target_dims is not None:
                x = x[:, :, config.target_dims]
                y = y[:, :, config.target_dims]
            if y.ndim == 3:
                y = y.squeeze(1)

            # Extract last reconstruction only
            # 重建后的数据，只取最后时刻点的一条记录 torch.Size([190, 15, 1]) ——>torch.Size([190, 1])
            recons.append(window_recon[:, -1, :].detach().cpu().numpy())
            predicts.append(window_predict.detach().cpu().numpy())

            test_label.append(attack_labels.cpu().numpy())
            test_data.append(y.cpu().numpy())

    return recons, predicts, test_data, test_label


def main(args):
    config = args

    stgat = args.stgat

    # # 实例化模型
    # stgat = STGAT(
    #     config.n_features,
    #     config.slide_win,
    #     config.out_dim,
    #     kernel_size=config.kernel_size,
    #     layer_numb=config.layer_numb,
    #     lstm_n_layers=config.lstm_n_layers,
    #     lstm_hid_dim=config.lstm_hid_dim,
    #     recon_n_layers=config.recon_n_layers,
    #     recon_hid_dim=config.recon_hid_dim,
    #     dropout=config.dropout,
    #     alpha=config.alpha
    # ).to(config.device)
    # # think: 每日训练模型时初始化，还是一直训练一个模型？思考ARIMA的话，是每次训练时都初始化

    # 设置工作目录
    save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log = []
    best_loss = 10000
    best_predict = []
    gt = []
    for epoch in range(config.epoch):
        train_loss = train(config, stgat, config.train_dataloader)  # done: 现在的dataloader传入的传入的不对，需要修正
        recons, predicts, test_data, test_label = test(config, stgat, config.test_dataloader)
        recons = np.concatenate(recons, axis=0)
        predicts = np.concatenate(predicts, axis=0)
        test_data = np.concatenate(test_data, axis=0)
        recons_loss = np.sqrt((recons - test_data) ** 2).mean(axis=1)
        predicts_loss = np.sqrt((predicts - test_data) ** 2).mean(axis=1)
        test_label = np.concatenate(test_label, axis=0)
        test_loss = predicts_loss.mean()
        if best_loss > test_loss:
            best_loss = test_loss
            # print(predicts.shape)
            if config.condition_control:
                best_predict = config.scaler.inverse_transform(np.repeat(predicts, config.n_features-1, axis=1))
                gt = config.scaler.inverse_transform(np.repeat(test_data, config.n_features-1, axis=1))
            else:
                best_predict = config.scaler.inverse_transform(np.repeat(predicts, config.n_features, axis=1))
                gt = config.scaler.inverse_transform(np.repeat(test_data, config.n_features, axis=1))
        log.append(
            {'epoch': epoch + 1, 'recons_loss': float(recons_loss.mean()), 'predicts_loss': float(predicts_loss.mean()),
             }
        )
    return best_loss, best_predict, gt


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    args.dataset = "ZX"
    args.group = "computer-b0503-01"

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # 设置运行的GPU
    device = torch.device('cuda')
    args.device = device

    # 设置数据集的维度和拟合的目标维度
    args.n_features = get_dim(args)
    target_dims = get_target_dims(args.dataset)
    if target_dims is None:
        out_dim = args.n_features
        print(f"Will forecast and reconstruct all {args.n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)
    args.target_dims = target_dims
    args.out_dim = out_dim
    print(args)

    (train_data, train_label), (test_data, test_label) = get_data_from_source(args)
    data = np.concatenate([train_data, test_data], axis=0)
    labels = np.concatenate([train_label, test_label], axis=0)
    args.n_features = data.shape[1]
    if args.condition_control:
        args.n_features = data.shape[1]+1
    train_scope = 288
    args.epoch = 30
    predict_h_all = []
    gt_h_all = []
    args.slide_stride = 12
    pre_gap = args.pre_gap
    pre_times = int(args.slide_stride / pre_gap)

    # 实例化模型
    stgat = STGAT(
        args.n_features,
        args.slide_win,
        args.out_dim,
        kernel_size=args.kernel_size,
        layer_numb=args.layer_numb,
        lstm_n_layers=args.lstm_n_layers,
        lstm_hid_dim=args.lstm_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    ).to(args.device)
    # think: 每日训练模型时初始化，还是一直训练一个模型？思考ARIMA的话，是每次训练时都初始化
    args.stgat = stgat

    for j in tqdm(range(train_scope * 2, len(data), train_scope)):
        (args.train, args.train_label), (args.test, args.test_label) = (data[j - train_scope * 2:j - train_scope],
                                                                        labels[j - train_scope * 2:j - train_scope]), (
                                                                           data[j - train_scope:j],
                                                                           labels[j - train_scope:j])
        predict_h_day = []
        gt_h_day = []
        for pre_t in range(1, args.slide_stride + 1, pre_gap):  # 预测未来一小时内三个时刻的值
            args.pre_term = pre_t  # 指定预测间隔
            args.train_dataloader, args.test_dataloader = data_load_from_exist_np(args)
            loss, predict_h, gt_h = main(args)
            print(f"predict_h.shape: {predict_h.shape}")
            predict_h_day.append(predict_h)
            gt_h_day.append(gt_h)
        predict_h_day_nd = np.array(predict_h_day)  # predict_h_day_nd.shape = (pre_times,24,k)
        gt_h_day_nd = np.array(gt_h_day)
        shape = predict_h_day_nd.shape
        predict_h_all.append(predict_h_day_nd.transpose(1, 2, 0))  # predict_h_all.shape = (days,24,k,pre_times)
        gt_h_all.append(gt_h_day_nd.transpose(1, 2, 0))
        print(f"predict_h_day_nd.shape: {predict_h_day_nd.shape}")
    scope_predicts = np.concatenate(predict_h_all, axis=0)
    scope_test_data = np.concatenate(gt_h_all, axis=0)
    scope_mse = up_down_mse_for_hour(scope_predicts[:, target_dims[0], :].squeeze(),
                                     scope_test_data[:, target_dims[0], :].squeeze())
    point_mse = metrics.mean_squared_error(scope_predicts.ravel(), scope_test_data.ravel())
    # done: change the position of saving predict and ground truth
    save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
    np.save(f"{save_path}/inner_gts_{args.target_dims[0]}_hour_{pre_gap}.npy", scope_test_data)
    np.save(f"{save_path}/best_predict_{args.target_dims[0]}_hour_{pre_gap}.npy", scope_predicts)
    print(f"scope_test_data.shape: {scope_test_data.shape}")
    print(f"scope_mse: {scope_mse}")
    results = {"scope_mse": float(scope_mse), "point_mse": float(point_mse)}
    if args.condition_control:
        with open(f'{save_path}/results_scope_{pre_gap}.json', "w") as f:
            json.dump(results, f, indent=2)
    else:
        with open(f'{save_path}/results_scope_{pre_gap}_False.json',"w") as f:
            json.dump(results, f, indent=2)
    # print(args)

# todo 将功率限制的label换成比例，然后再看看效果
