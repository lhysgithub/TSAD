#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
            # 将输入数据处理成与重建输出相同的形状,如果为NASA数据集则只取第一个维度的数据
            x = x[:, :, config.target_dims]
            # 将输入数据处理成与预测输出相同的形状,如果未NASA数据集则只取第一个维度的数据
            y = y[:, :, config.target_dims]
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

    # 设置工作目录
    save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log = []
    best_loss = 10000
    best_predict = []
    gt = []
    for epoch in range(config.epoch):
        if not config.stop_train:
            stgat.train()
            # done: 现在的dataloader传入的传入的不对，需要修正
            train_loss = train(config, stgat, config.train_dataloader)
        stgat.eval()
        recons, predicts, test_data, test_label = test(
            config, stgat, config.test_dataloader)
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
                best_predict = config.scaler.inverse_transform(
                    np.repeat(predicts, config.n_features-1, axis=1))
                gt = config.scaler.inverse_transform(
                    np.repeat(test_data, config.n_features-1, axis=1))
            else:
                best_predict = config.scaler.inverse_transform(
                    np.repeat(predicts, config.n_features, axis=1))
                gt = config.scaler.inverse_transform(
                    np.repeat(test_data, config.n_features, axis=1))
        log.append(
            {'epoch': epoch + 1, 'recons_loss': float(recons_loss.mean()), 'predicts_loss': float(predicts_loss.mean()),
             }
        )
    return best_loss, best_predict, gt


def daily_trainer(args):
    # 如果实体感知，可以对于每个实体初始化一个模型
    if args.entity_wised_model:
        args.stgat = STGAT(
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

    # 对于每个实体，重新读取数据
    (train_data, train_label), (test_data, test_label) = get_data_from_source(args)
    data = np.concatenate([train_data, test_data], axis=0)
    labels = np.concatenate([train_label, test_label], axis=0)
    tt = data.T

    # 模拟每天连续训练模型
    day = 0
    predict_h_all = []
    gt_h_all = []
    for j in tqdm(range(train_scope * 3, len(data), train_scope)):
        (args.train, args.train_label), (args.test, args.test_label) = (
            data[j - train_scope * 3:j - train_scope], labels[j - train_scope * 3:j - train_scope]), (data[j - train_scope-args.slide_win:j], labels[j - train_scope-args.slide_win:j])
        day += 1
        tt = args.test.T

        # 对于每个实体，考虑是否部分数据只用于测试
        if day > args.stop_train_days:
            args.stop_train = True
        else:
            args.stop_train = False

        predict_h_day = []
        gt_h_day = []
        # 预测未来时间段内的全部时刻（此处使用的是对于每个时刻分别训练一个模型）
        # todo 待优化，让模型直接预测未来一段时间内全部时刻的值
        for pre_t in range(1, args.max_pre_term + 1, args.pre_gap):  # 预测未来一小时12个时刻的值
            args.pre_term = pre_t  # 指定预测间隔
            args.train_dataloader, args.test_dataloader = data_load_from_exist_np(
                args)
            loss, predict_h, gt_h = main(args)
            # print(f"predict_h.shape: {predict_h.shape}")
            predict_h_day.append(predict_h)
            gt_h_day.append(gt_h)
        # predict_h_day_nd.shape = (pre_times,24,k)
        predict_h_day_nd = np.array(predict_h_day)
        gt_h_day_nd = np.array(gt_h_day)
        shape = predict_h_day_nd.shape

        # 此处考虑是否值记录训练集未见过的测试数据
        if args.only_plot_untrain:
            if day <= args.stop_train_days:
                continue
        # predict_h_all.shape = (days,24,k,pre_times)
        predict_h_all.append(predict_h_day_nd.transpose(1, 2, 0))
        gt_h_all.append(gt_h_day_nd.transpose(1, 2, 0))
        # print(f"predict_h_day_nd.shape: {predict_h_day_nd.shape}")
    return predict_h_all, gt_h_all


def save_data_and_evaluation(args, predict_h_all, gt_h_all, file_name):
    # 整理得到的预测和真实数据
    scope_predicts = np.concatenate(predict_h_all, axis=0)
    scope_test_data = np.concatenate(gt_h_all, axis=0)
    save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
    np.save(
        f"{save_path}/inner_gts_{args.target_dims[0]}_hour.npy", scope_test_data)
    np.save(
        f"{save_path}/best_predict_{args.target_dims[0]}_hour.npy", scope_predicts)
    print(f"scope_test_data.shape: {scope_test_data.shape}")

    # 评估
    plot_multi_curve(args, save_path, 2, f"zx_stgat_hour_{args.condition_control}_{file_name}", args.target_dims[0], list(
        range(0, 12, 1)), "stgat")
    point_mse, scope_mse = evaluate_multi_curve(
        args, save_path, 2, f"zx_stgat_hour_{args.condition_control}_{file_name}", args.target_dims[0], list(range(0, 12, 1)), "stgat")
    results = {"scope_mse": float(scope_mse), "point_mse": float(point_mse)}

    # 存储结果
    with open(f'{save_path}/results_scope_{args.condition_control}_{file_name}.json', "w") as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    args.dataset = "ZX"
    args.group = "computer-b0503-01"
    # args.condition_control = False
    
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
        print(
            f"Will forecast and reconstruct all {args.n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)
    args.target_dims = target_dims
    args.out_dim = out_dim
    print(args)

    # 运行参数初始化
    (train_data, train_label), (test_data, test_label) = get_data_from_source(args)
    data = np.concatenate([train_data, test_data], axis=0)
    labels = np.concatenate([train_label, test_label], axis=0)
    args.n_features = data.shape[1]
    if args.condition_control:
        args.n_features = data.shape[1]+1
    train_scope = 288
    # args.epoch = 2
    predict_h_all = []
    gt_h_all = []
    # args.slide_win = 12
    args.slide_stride = 12
    # args.slide_stride = 24
    args.max_pre_term = 12
    pre_gap = args.pre_gap
    pre_times = int(args.max_pre_term / pre_gap)
    shape = 2
    plt.figure(figsize=(8 * shape, 6 * shape))
    
    # 提前评估
    # file_name = "test"
    # save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
    # plot_multi_curve(args, save_path, 2, f"zx_stgat_hour_{args.condition_control}_{file_name}", args.target_dims[0], list(
    #     range(0, 12, 1)), "stgat")
    # point_mse, scope_mse = evaluate_multi_curve(
    #     args, save_path, 2, f"zx_stgat_hour_{args.condition_control}_{file_name}", args.target_dims[0], list(range(0, 12, 1)), "stgat")

    # 实例化模型
    args.stgat = STGAT(
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
    # think: 每日训练模型时初始化，还是一直训练一个模型？思考ARIMA的话，是每次训练时都初始化。但是STGAT这种神经网络模型需要更多的训练数据，所以需要持续学习。
    # for m in args.stgat.modules():
    #     if isinstance(m, torch.nn.Conv2d):
    #         torch.nn.kaiming_normal_(m.weight, mode = 'fan_in')
    # args.multi_entities_train = True
    if args.multi_entities_train:
        # 多实体训练
        host_list = ["computer-b0503-03"]
        # host_list = ["computer-b0503-02", "computer-b0503-03"]
        for args.group in host_list:
            predict_h_all = []
            gt_h_all = []
            predict_h_all_, gt_h_all_ = daily_trainer(args)
            predict_h_all = predict_h_all + predict_h_all_
            gt_h_all = gt_h_all + gt_h_all_
            save_data_and_evaluation(
                args, predict_h_all, gt_h_all, f"{args.slide_win}_{args.epoch}_origin")
    
    if args.only_plot_untrain:
        stop_train_days = [16,20,22]
        args.group = "computer-b0503-03"
        for args.stop_train_days in stop_train_days:
            predict_h_all = []
            gt_h_all = []
            predict_h_all_, gt_h_all_ = daily_trainer(args)
            predict_h_all = predict_h_all + predict_h_all_
            gt_h_all = gt_h_all + gt_h_all_
            save_data_and_evaluation(
                args, predict_h_all, gt_h_all, f"{args.slide_win}_{args.epoch}_stop_train_{args.stop_train_days}")

    # 跨实体测试
    args.cross_entity = False
    if args.cross_entity:
        args.stop_train_days = -1
        predict_h_all = []
        gt_h_all = []
        host_list = ["computer-b0503-01"]
        for args.group in host_list:
            predict_h_all = []
            gt_h_all = []
            predict_h_all_, gt_h_all_ = daily_trainer(args)
            predict_h_all = predict_h_all + predict_h_all_
            gt_h_all = gt_h_all + gt_h_all_
            save_data_and_evaluation(
                args, predict_h_all, gt_h_all, "multi")

    # 多条件变量预测
    # args.condition_infer = True
    if args.condition_infer:
        args.stop_train_days = -1
        predict_h_all = []
        gt_h_all = []
        host_list = ["computer-b0503-03"]
        variate_list = [0, 10, 20, 30, 40]
        for args.group in host_list:
            for args.condition_variate in variate_list:
                predict_h_all = []
                gt_h_all = []
                predict_h_all_, gt_h_all_ = daily_trainer(args)
                predict_h_all = predict_h_all + predict_h_all_
                gt_h_all = gt_h_all + gt_h_all_
                save_data_and_evaluation(
                    args, predict_h_all, gt_h_all, f"{args.slide_win}_{args.epoch}_{args.condition_variate}")

    # for args.slide_win in [12]:
    # for args.slide_win in [12,24,48,72,96]:
    #     for args.epoch in [1,2,4,6,8]: