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


def train(config, model, train_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    forecast_criterion = nn.MSELoss().to(config.device)
    recon_criterion = nn.MSELoss().to(config.device)
    losses = []
    model.train()
    # check fc tc edge 是否有问题？ lhy 没问题
    # print(len(train_dataloader.dataset))
    for x, y, attack_labels, fc_edge_index, tc_edge_index in tqdm(train_dataloader):
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
    # print(len(test_dataloader.dataset))
    with torch.no_grad():
        for x, y, attack_labels, fc_edge_index, tc_edge_index in tqdm(test_dataloader):
            x, y, fc_edge_index, tc_edge_index = [item.float().to(config.device) for item in
                                                  [x, y, fc_edge_index, tc_edge_index]]
            y = y.unsqueeze(1)
            # Shifting input to include the observed value (y) when doing the reconstruction
            recon_x = torch.cat((x[:, 1:, :], y), dim=1)
            window_recon, _ = model(recon_x, fc_edge_index, tc_edge_index)
            _, window_predict = model(x,fc_edge_index,tc_edge_index)

            if config.target_dims is not None:
                x = x[:, :, config.target_dims]
                y = y[:, :, config.target_dims]
            if y.ndim == 3:
                y = y.squeeze(1)

            # Extract last reconstruction only
            recons.append(window_recon[:, -1,:].detach().cpu().numpy())  # 重建后的数据，只取最后时刻点的一条记录 torch.Size([190, 15, 1]) ——>torch.Size([190, 1])
            predicts.append(window_predict.detach().cpu().numpy())

            test_label.append(attack_labels.cpu().numpy())
            test_data.append(y.cpu().numpy())

    return recons, predicts, test_data, test_label


def main(args):
    config = args

    # 实例化模型
    stgat = STGAT(
        config.n_features,
        config.slide_win,
        config.out_dim,
        kernel_size=config.kernel_size,
        layer_numb=config.layer_numb,
        lstm_n_layers=config.lstm_n_layers,
        lstm_hid_dim=config.lstm_hid_dim,
        recon_n_layers=config.recon_n_layers,
        recon_hid_dim=config.recon_hid_dim,
        dropout=config.dropout,
        alpha=config.alpha
    ).to(config.device)

    # 设置工作目录
    save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log = []
    best_loss = 10000
    best_predict = []
    gt = []
    for epoch in range(config.epoch):
        train_loss = train(config, stgat, train_dataloader)
        print(f"[train epoch {epoch}] train loss: {train_loss:.4f}")
        recons, predicts, test_data, test_label = test(config, stgat, test_dataloader)
        recons = np.concatenate(recons, axis=0)
        predicts = np.concatenate(predicts, axis=0)
        test_data = np.concatenate(test_data, axis=0)
        recons_loss = np.sqrt((recons - test_data) ** 2).mean(axis=1)
        predicts_loss = np.sqrt((predicts - test_data) ** 2).mean(axis=1)
        test_label = np.concatenate(test_label, axis=0)
        test_loss = predicts_loss.mean()
        if best_loss > test_loss:
            best_loss = test_loss
            # best_predict = predicts
            # gt = test_data
            # best_predict = config.scaler.inverse_transform(predicts)
            # gt = config.scaler.inverse_transform(test_data)
            best_predict = config.scaler.inverse_transform(np.repeat(predicts,config.n_features,axis=1))
            gt = config.scaler.inverse_transform(np.repeat(test_data,config.n_features,axis=1))
        # bf_eval = bf_search(recons_loss, test_label, start=0.01, end=args.confidence, step_num=100, verbose=False)
        # test_loss = np.array(recons_loss).mean()
        print(f'[test epoch {epoch}] recons loss: {recons_loss.mean():.4f} | predicts loss: {predicts_loss.mean():.4f} '
              # f'| best F1: {bf_eval["f1"]:.4f} | precision: '
              # f'{bf_eval["precision"]:.4f} | Recall: {bf_eval["recall"]:.4f} '
              )
        log.append({'epoch': epoch + 1, 'recons_loss': float(recons_loss.mean()), 'predicts_loss' : float(predicts_loss.mean()),
                    # 'precision': float(bf_eval["precision"]),
                    # 'recall': float(bf_eval["recall"]), 'f1': float(bf_eval["f1"]),'TP':int(bf_eval["TP"]),
                    # 'TN':int(bf_eval["TN"]),'FP':int(bf_eval["FP"]),'FN':int(bf_eval["FN"])
                    }
                   )

    np.save(f"{save_path}/inner_gts_{config.pre_term}_term_{config.target_dims[0]}.npy", gt)
    np.save(f"{save_path}/best_predict_{config.pre_term}_term_{config.target_dims[0]}.npy", best_predict)

    # 记录运行结果
    with open(f"{save_path}/summary_{config.pre_term}_{config.target_dims[0]}.txt", "w") as f:
        bestId = 0
        for i in range(len(log)):
            if log[i]["predicts_loss"] < log[bestId]["predicts_loss"]:
                bestId = i
        json.dump(log[bestId], f, indent=2)
        print(f"best predicts_loss: {log[bestId]['predicts_loss']}")

    return best_loss

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

    loss_log = []
    for i in tqdm(range(1,13)):
        args.pre_term = i
        # 加载数据
        train_dataloader, test_dataloader = data_load(args)
        loss = main(args)
        loss_log.append(float(loss))

    save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
    with open(f"{save_path}/summary_term_{args.target_dims[0]}.txt", "w") as f:
        bestId = 0
        json.dump(loss_log, f, indent=2)
        print(f"predicts_losses: {loss_log}")