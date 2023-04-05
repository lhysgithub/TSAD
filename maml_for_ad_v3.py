import argparse
import time
import os
import typing

import pandas as pd
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('bmh')

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import higher

from support.omniglot_loaders import OmniglotNShot
from mtad_gat import MTAD_GAT
from mtad_gat_norm import MTAD_GAT_norm
from args import get_parser
from timeSeriesDatabase_v2 import TimeSeriesDatabase
from sklearn.metrics import roc_curve, auc, precision_recall_curve, mean_squared_error, f1_score, \
    precision_recall_fscore_support, confusion_matrix, precision_score, recall_score, roc_auc_score
from eval_methods import bf_search
import json
from utils import *
from tqdm import tqdm


def main():
    parser = get_parser()
    args = parser.parse_args()

    sum_path = f"output/{args.dataset}/{args.group}/{args.save_dir}/summary.txt"
    if os.path.exists(sum_path):
        f1_ = get_f1_for_maml(sum_path)
        if args.save_dir != "temp" and 0.99 >= f1_ >= 0.01:
            return

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

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

    # 获取数据集
    db = TimeSeriesDatabase(args)

    # 获取网络
    net = MTAD_GAT(n_features=args.n_features,
                   window_size=args.lookback,
                   out_dim=args.out_dim,
                   kernel_size=args.kernel_size,
                   use_gatv2=args.use_gatv2,
                   feat_gat_embed_dim=args.feat_gat_embed_dim,
                   time_gat_embed_dim=args.time_gat_embed_dim,
                   gru_n_layers=args.gru_n_layers,
                   gru_hid_dim=args.gru_hid_dim,
                   forecast_n_layers=args.fc_n_layers,
                   forecast_hid_dim=args.fc_hid_dim,
                   recon_n_layers=args.recon_n_layers,
                   recon_hid_dim=args.recon_hid_dim,
                   dropout=args.dropout,
                   alpha=args.alpha).to(device)
    if args.norm_model == "norm":
        net = MTAD_GAT_norm(n_features=args.n_features,
                            window_size=args.lookback,
                            out_dim=args.out_dim,
                            kernel_size=args.kernel_size,
                            use_gatv2=args.use_gatv2,
                            feat_gat_embed_dim=args.feat_gat_embed_dim,
                            time_gat_embed_dim=args.time_gat_embed_dim,
                            gru_n_layers=args.gru_n_layers,
                            gru_hid_dim=args.gru_hid_dim,
                            forecast_n_layers=args.fc_n_layers,
                            forecast_hid_dim=args.fc_hid_dim,
                            recon_n_layers=args.recon_n_layers,
                            recon_hid_dim=args.recon_hid_dim,
                            dropout=args.dropout,
                            alpha=args.alpha).to(device)

    # 设置工作目录
    save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 设置优化器
    meta_opt = optim.Adam(net.parameters(), lr=1e-3)

    # 运行训练和测试
    best_f1 = 0
    with torch.backends.cudnn.flags(enabled=False):
        train_log = []
        test_log = []
        for epoch in range(args.epochs):
            train(args, db, net, device, meta_opt, epoch, train_log)
            recons, attentions, inner_gt, inter_gt = test(args, db, net, device, meta_opt, epoch, test_log)
            if test_log[-1]["f1"] > best_f1:
                torch.save(net.state_dict(), f"{save_path}/best_model.pt")
                np.save(f"{save_path}/best_recons.npy", recons)
                sum_attentions = attentions.sum(axis=1)
                np.save(
                    f"{save_path}/best_attentions_{args.open_maml}_data_enhancement_{args.using_labeled_val}_semi_all.npy",
                    attentions)
            if test_log[-1]["f1"] < 0.01 or test_log[-1]["f1"] > 0.99:
                break
        np.save(f"{save_path}/inner_gts.npy", inner_gt)
        np.save(f"{save_path}/inter_gts.npy", inter_gt)

    # 记录运行结果
    with open(f"{save_path}/summary.txt", "w") as f:
        bestId = 0
        for i in range(len(test_log)):
            if test_log[i]["f1"] > test_log[bestId]["f1"]:
                bestId = i
        json.dump(test_log[bestId], f, indent=2)
        print(f"best test f1: {test_log[bestId]['f1']}")


def train(args, db, net, device, meta_opt, epoch, log):
    net.train()
    n_test_iter = 1
    for batch_idx in range(n_test_iter):
        start_time = time.time()
        spt_loader, qry_loader = db.next('train')
        meta_opt.zero_grad()
        inner_opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        meta_opt.zero_grad()
        spt_losses = [0]
        pres = []
        recons = []
        inner_gt = []
        inter_gt = []
        qry_losses = [0]
        recon_pre_losses = [0]
        attentions = []
        with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
            fnet.train()
            for row, x, z, y, s in tqdm(spt_loader):
                train_loss1 = spt_forward(row, x, z, y, s, args, fnet, diffopt, db)
                spt_losses.append(train_loss1)
            print(f"[Epoch {epoch + 1}] train_loss: {np.array(spt_losses).mean()}")

            para_dict = {}
            for name, parms in fnet.named_parameters():
                para_dict[name] = parms
            for name, parms in net.named_parameters():
                parms.data = parms + 1 * (para_dict[name] - parms)

            if args.using_labeled_val:
                net.train()
                for row, x, z, y, s in tqdm(qry_loader):
                    test_loss, recon, pre, inner_gt_, inter_gt_, attentions_ = qry_forward(x, z, y, args, net, meta_opt)
                    qry_losses.append(test_loss)
                    recons.append(recon)
                    pres.append(pre)
                    inner_gt.append(inner_gt_)
                    inter_gt.append(inter_gt_)
                    attentions.append(attentions_)

                attentions = np.concatenate(attentions, axis=0)
                pres = np.concatenate(pres, axis=0)
                recons = np.concatenate(recons, axis=0)
                inner_gt = np.concatenate(inner_gt, axis=0)
                inter_gt = np.concatenate(inter_gt, axis=0)
                # anomaly_scores = np.sqrt((recons - inner_gt) ** 2) + np.sqrt((pres - inner_gt) ** 2)
                # anomaly_scores = np.sqrt((recons - inner_gt) ** 2)
                anomaly_scores = np.sqrt((pres - inner_gt) ** 2)
                anomaly_scores = np.mean(anomaly_scores, axis=1)  # 此处使用的是mean，那么前边的损失也需要使用mean！
                bf_eval = bf_search(anomaly_scores, inter_gt, start=0.01, end=args.confidence,
                                    step_num=int(args.confidence / 0.01),
                                    verbose=False)
                train_mean_loss = np.array(spt_losses).mean()
                test_mean_loss = np.array(qry_losses).mean()
                test_recon_pre_loss = np.array(recon_pre_losses).mean()
                iter_time = time.time() - start_time
                di = print_info(epoch, train_mean_loss, test_mean_loss, test_recon_pre_loss, bf_eval, iter_time)
                log.append(di)
                return recons, attentions, inner_gt, inter_gt


def test(args, db, net, device, meta_opt, epoch, log):
    net.train()
    n_test_iter = 1
    for batch_idx in range(n_test_iter):
        start_time = time.time()
        spt_loader, qry_loader = db.next('test')
        meta_opt.zero_grad()
        inner_opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        meta_opt.zero_grad()
        spt_losses = [0]
        pres = []
        recons = []
        inner_gt = []
        inter_gt = []
        qry_losses = []
        recon_pre_losses = [0]
        attentions = []
        # with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
        #     for row, x, z, y, s in tqdm(spt_loader):
        #         train_loss1 = spt_forward(row, x, z, y, s, args, fnet, diffopt, db, "test")
        #         spt_losses.append(train_loss1)
        #
        #     para_dict = {}
        #     for name, parms in fnet.named_parameters():
        #         para_dict[name] = parms
        #     for name, parms in net.named_parameters():
        #         parms.data = parms + 1 * (para_dict[name] - parms)

        net.eval()
        with torch.no_grad():
            for row, x, z, y, s in tqdm(qry_loader):
                test_loss, recon, pre, inner_gt_, inter_gt_, attentions_ = qry_forward(x, z, y,
                                                                                       args, net, meta_opt, "test")
                qry_losses.append(test_loss)
                recons.append(recon)
                pres.append(pre)
                inner_gt.append(inner_gt_)
                inter_gt.append(inter_gt_)
                attentions.append(attentions_)

        attentions = np.concatenate(attentions, axis=0)
        pres = np.concatenate(pres, axis=0)
        recons = np.concatenate(recons, axis=0)
        inner_gt = np.concatenate(inner_gt, axis=0)
        inter_gt = np.concatenate(inter_gt, axis=0)
        # anomaly_scores = np.sqrt((pres - inner_gt) ** 2) + np.sqrt((recons - inner_gt) ** 2)
        # anomaly_scores = np.sqrt((recons - inner_gt) ** 2)
        anomaly_scores = np.sqrt((pres - inner_gt) ** 2)
        anomaly_scores = np.mean(anomaly_scores, axis=1)  # 此处使用的是mean，那么前边的损失也需要使用mean！
        bf_eval = bf_search(anomaly_scores, inter_gt, start=0.01, end=args.confidence,
                            step_num=int(args.confidence / 0.01), verbose=False)
        train_mean_loss = np.array(spt_losses).mean()
        test_mean_loss = np.array(qry_losses).mean()
        test_recon_pre_loss = np.array(recon_pre_losses).mean()
        iter_time = time.time() - start_time
        di = print_info(epoch, train_mean_loss, test_mean_loss, test_recon_pre_loss, bf_eval, iter_time)
        log.append(di)
        return recons, attentions, inner_gt, inter_gt


def spt_forward(row, x, z, y, s, args, model, opt, db, mode="train"):
    row, x, z, y, s = [(item).float().to(args.device) for item in
                       [row, x, z, y, s]]
    # z = z.unsqueeze(1)
    if np.random.random() < args.r1:
        x = row
    preds, spt_logits = model(x)
    if preds.ndim == 3:
        preds = preds.squeeze(1)
    if z.ndim == 3:
        z = z.squeeze(1)
    if args.target_dims is not None:
        row = row[:, :, args.target_dims]
        x = x[:, :, args.target_dims]
        z = z[:, args.target_dims]
        spt_logits = spt_logits[:, :, args.target_dims]
        preds = preds[:, args.target_dims]
    # if mode == "train":
    #     spt_loss = 0
    #     for i in range(len(s)):
    #         sd = bool_list2list(s[i])
    #         xi = x[i,:,sd]
    #         zi = z[i,sd]
    #         spti = spt_logits[i,:,sd]
    #         pi = preds[i,sd]
    #         spt_loss += torch.sqrt(F.mse_loss(spti, xi)) + torch.sqrt(F.mse_loss(pi, zi))
    # else:
    #     spt_loss = torch.sqrt(F.mse_loss(spt_logits, x)) + torch.sqrt(F.mse_loss(preds, z))
    # spt_loss = torch.sqrt(F.mse_loss(spt_logits, row)) + torch.sqrt(F.mse_loss(preds, z))
    spt_loss = torch.sqrt(F.mse_loss(preds, z))
    opt.step(spt_loss)
    return spt_loss.item()


def qry_forward(x, z, y, args, model, opt, mode="train"):
    x, z, y = [(item).float().to(args.device) for item in
               [x, z, y]]
    # z = z.unsqueeze(1)
    opt.zero_grad()
    x_hat = torch.cat((x[:, 1:, :], z), dim=1)
    pre_logits, original_recon = model(x)
    _, qry_logits = model(x_hat)
    attention = model.get_gat_attention(x_hat)
    recon = qry_logits[:, -1, :]
    pre = pre_logits
    z_hat = z.squeeze(dim=1)
    y_hat = y.squeeze(dim=1)
    if args.target_dims != None:
        z_hat = z_hat[:, args.target_dims]
    # qry_loss = torch.sqrt((recon - z_hat) ** 2) + torch.sqrt((pre - z_hat) ** 2)
    # qry_loss = torch.sqrt((recon - z_hat) ** 2)
    qry_loss = torch.sqrt((pre - z_hat) ** 2)
    qry_loss = qry_loss.mean(dim=1)
    # qry_loss = F.mse_loss(qry_loss, y_hat * args.confidence)
    # qry_loss = torch.multiply(qry_loss, y_hat * args.confidence * -1).mean()
    qry_loss = torch.multiply(qry_loss, ((y_hat * -2) + torch.ones_like(y_hat)) * args.confidence).mean()
    if mode == "train" and args.using_labeled_val:
        qry_loss.backward()
        opt.step()
    inner_gt = z_hat.detach().cpu().numpy()
    inter_gt = y_hat.detach().cpu().numpy()
    recons = recon.detach().cpu().numpy()
    attentions = attention.detach().cpu().numpy()
    pres = pre.detach().cpu().numpy()
    return qry_loss.item(), recons, pres, inner_gt, inter_gt, attentions


if __name__ == '__main__':
    main()
