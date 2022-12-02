#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example shows how to use higher to do Model Agnostic Meta Learning (MAML)
for few-shot Omniglot classification.
For more details see the original MAML paper:
https://arxiv.org/abs/1703.03400

This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py

Our MAML++ fork and experiments are available at:
https://github.com/bamos/HowToTrainYourMAMLPytorch
"""

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
from args import get_parser
from timeSeriesDatabase import TimeSeriesDatabase
from sklearn.metrics import roc_curve, auc, precision_recall_curve, mean_squared_error, f1_score, precision_recall_fscore_support,confusion_matrix, precision_score, recall_score, roc_auc_score
from eval_methods import bf_search
import json
from utils import *


def main():
    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda')
    args.device = device
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
    db = TimeSeriesDatabase(args)
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

    save_path = f"output/{args.dataset}/{args.group}/maml"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    meta_opt = optim.Adam(net.parameters(), lr=1e-3)
    with torch.backends.cudnn.flags(enabled=False):
        log = []
        for epoch in range(5):
            # train(args,db, net, device, meta_opt, epoch, log)
            test(args,db, net, device, meta_opt, epoch, log)
            # plot(log)
            with open(f"{save_path}/summary.txt", "w") as f:
                json.dump(log[-1], f, indent=2)


def train(args,db, net, device, meta_opt, epoch, log):
    net.train()
    n_train_iter = 5
    for batch_idx in range(n_train_iter):
        start_time = time.time()
        spt_loader, qry_loader = db.next('train')
        inner_opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        meta_opt.zero_grad()
        with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
            fnet.train()
            train_losses = []
            for x, z, y in spt_loader:
                x = x.to(device)
                y = y.to(device)
                z = z.to(device)
                preds, spt_logits = fnet(x)
                if args.target_dims is not None:
                    x = x[:, :, args.target_dims]
                    z = z[:, :, args.target_dims].squeeze(-1)
                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if z.ndim == 3:
                    z = z.squeeze(1)
                spt_loss = torch.sqrt(F.mse_loss(spt_logits, x)) + torch.sqrt(F.mse_loss(preds, z))
                diffopt.step(spt_loss)
                train_losses.append(spt_loss.item())

            pres = []
            recons = []
            inner_gt = []
            inter_gt = []
            qry_losses = []
            for x, z, y in qry_loader:
                x = x.to(device)
                y = y.to(device)
                z = z.to(device)
                meta_opt.zero_grad()  # todo test is it work？
                x_hat = torch.cat((x[:, 1:, :], z), dim=1)
                pre_logits, _ = fnet(x)
                _, qry_logits = fnet(x_hat)
                recon = qry_logits[:, -1, :]
                pre = pre_logits
                recons.append(recon.detach().cpu().numpy())
                pres.append(pre.detach().cpu().numpy())
                z_hat = z.squeeze(dim=1)
                y_hat = y.squeeze(dim=1)
                inner_gt.append(z_hat.detach().cpu().numpy())
                inter_gt.append(y_hat.detach().cpu().numpy())

                qry_loss = (recon - z_hat) ** 2 + (pre - z_hat) ** 2
                qry_loss = torch.sigmoid(qry_loss.mean(dim=1))
                qry_loss = F.mse_loss(qry_loss, y_hat)
                qry_loss.backward()
                # meta_opt.step() # todo check is it work
                qry_losses.append(qry_loss.item())
            meta_opt.step()  # todo check is it work

            pres = np.concatenate(pres, axis=0)
            recons = np.concatenate(recons, axis=0)
            inner_gt = np.concatenate(inner_gt, axis=0)
            inter_gt = np.concatenate(inter_gt, axis=0)
            anomaly_scores = np.zeros_like(inner_gt)
            for i in range(pres.shape[1]):
                a_score = np.sqrt((pres[:, i] - inner_gt[:, i]) ** 2) + np.sqrt((recons[:, i] - inner_gt[:, i]) ** 2)
                anomaly_scores[:, i] = a_score
            anomaly_scores = np.mean(anomaly_scores, 1)
            bf_eval = bf_search(anomaly_scores, inter_gt, start=0.01, end=1, step_num=100, verbose=False)
            train_mean_loss = np.array(train_losses).mean()
            test_mean_loss = np.array(qry_losses).mean()
            iter_time = time.time() - start_time
            if batch_idx % 1 == 0:
                print(
                    f'[Train Epoch {epoch + 1:.2f}] | Train loss: {train_mean_loss:.2f} | '
                    f'Test Loss: {test_mean_loss:.2f} | best F1: {bf_eval["f1"]:.2f} | precision: '
                    f'{bf_eval["precision"]:.2f} | Recall: {bf_eval["recall"]:.2f} | Time: {iter_time:.2f}')
            log.append({
                'epoch': epoch + 1,
                'loss': test_mean_loss,
                'precision': bf_eval["precision"],
                'recall': bf_eval["recall"],
                'f1': bf_eval["f1"],
                'mode': 'train',
                'time': time.time(),
            })


def test(args,db, net, device, meta_opt, epoch, log):
    net.train()
    n_test_iter = 1
    for batch_idx in range(n_test_iter):
        spt_loader, qry_loader = db.next('test')
        inner_opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        meta_opt.zero_grad()
        with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
            fnet.train()
            train_losses = []
            for x, z, y in spt_loader:
                x = x.to(device)
                y = y.to(device)
                z = z.to(device)
                preds, spt_logits = fnet(x)
                if args.target_dims is not None:
                    x = x[:, :, args.target_dims]
                    z = z[:, :, args.target_dims].squeeze(-1)
                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if z.ndim == 3:
                    z = z.squeeze(1)
                spt_loss = torch.sqrt(F.mse_loss(spt_logits, x)) + torch.sqrt(F.mse_loss(preds, z))
                diffopt.step(spt_loss)
                train_losses.append(spt_loss.item())

            pres = []
            recons = []
            inner_gt = []
            inter_gt = []
            qry_losses = []
            for x, z, y in qry_loader:
                x = x.to(device)
                y = y.to(device)
                z = z.to(device)
                meta_opt.zero_grad()  # todo test is it work？
                x_hat = torch.cat((x[:, 1:, :], z), dim=1)
                pre_logits, _ = fnet(x)
                _, qry_logits = fnet(x_hat)
                recon = qry_logits[:, -1, :]
                pre = pre_logits
                recons.append(recon.detach().cpu().numpy())
                pres.append(pre.detach().cpu().numpy())
                z_hat = z.squeeze(dim=1)
                y_hat = y.squeeze(dim=1)
                inner_gt.append(z_hat.detach().cpu().numpy())
                inter_gt.append(y_hat.detach().cpu().numpy())

                qry_loss = (recon - z_hat) ** 2 + (pre - z_hat) ** 2
                qry_loss = torch.sigmoid(qry_loss.max(dim=1)[0])
                qry_loss = F.mse_loss(qry_loss, y_hat)
                qry_loss.backward()
                meta_opt.step() # todo check is it work
                qry_losses.append(qry_loss.item())
            # meta_opt.step()  # todo check is it work

            pres = np.concatenate(pres, axis=0)
            recons = np.concatenate(recons, axis=0)
            inner_gt = np.concatenate(inner_gt, axis=0)
            inter_gt = np.concatenate(inter_gt, axis=0)
            anomaly_scores = np.zeros_like(inner_gt)
            for i in range(pres.shape[1]):
                a_score = np.sqrt((pres[:, i] - inner_gt[:, i]) ** 2) + np.sqrt((recons[:, i] - inner_gt[:, i]) ** 2)
                anomaly_scores[:, i] = a_score
            anomaly_scores = np.mean(anomaly_scores, 1)
            bf_eval = bf_search(anomaly_scores, inter_gt, start=0.01, end=1, step_num=100, verbose=False)
            train_mean_loss = np.array(train_losses).mean()
            test_mean_loss = np.array(qry_losses).mean()
            # iter_time = time.time() - start_time
            if batch_idx % 1 == 0:
                print(
                    f'[Test Epoch {epoch + 1:.2f}] | Train loss: {train_mean_loss:.2f} | '
                    f'Test Loss: {test_mean_loss:.2f} | best F1: {bf_eval["f1"]:.2f} | precision: '
                    f'{bf_eval["precision"]:.2f} | Recall: {bf_eval["recall"]:.2f}')
            log.append({
                'epoch': epoch + 1,
                'loss': test_mean_loss,
                'precision': bf_eval["precision"],
                'recall': bf_eval["recall"],
                'f1': bf_eval["f1"],
                'mode': 'train',
                'time': time.time(),
            })


            # anomaly_predictions = []
            # ground_truths = []
            # anomaly_scores = []
            # with torch.no_grad():
            #     for j in range(qry_inner_batches):
            #         x = torch.from_numpy(x_qry[i][j]).to(device)
            #         y = torch.from_numpy(y_qry[i][j]).to(device)
            #         z = torch.from_numpy(z_qry[i][j]).to(device)
            #         x_hat = torch.cat((x[:,1:,:],z),dim=1)
            #         pre_logits, _ = net(x)
            #         _ , qry_logits = net(x_hat)
            #         window_rescon = qry_logits[:, -1, :].detach().cpu().numpy()
            #         window_prediction = pre_logits
            #
            #         qry_mse_loss = F.mse_loss(qry_logits, x_hat)
            #         qry_mes = torch.sqrt(torch.pow(qry_logits - x_hat, 2.0))
            #         qry_mes = torch.mean(qry_mes, dim=2)
            #         # qry_mes, _ = torch.max(qry_mes, dim=2)
            #         qry_mes = qry_mes + torch.sqrt(torch.pow(pre_logits - z,2.0))
            #         isNan = qry_mes[0][0].item()
            #         if isNan != isNan:
            #             continue
            #         anomaly_scores.append(qry_mes.detach().cpu().numpy())
            #         qry_mes = torch.sigmoid(qry_mes)
            #         qry_loss = F.mse_loss(qry_mes, y) + qry_mse_loss
            #         qry_acc = (qry_mes * y).sum().item() / qry_inner_batches
            #
            #         length = int(qry_mes.shape[1] * 0.10)
            #         vals, indices = qry_mes.topk(k=length, dim=1, sorted=True)
            #         kth_vals_column = vals[:, -1].reshape(-1, 1)
            #         kth_vals = kth_vals_column.repeat(1, qry_mes.shape[1])
            #         anomaly_prediction = torch.where(qry_mes > kth_vals, torch.ones_like(qry_mes),
            #                                          torch.zeros_like(qry_mes))
            #         anomaly_prediction_np = anomaly_prediction.detach().cpu().numpy()
            #         ground_truth = y.detach().cpu().numpy()
            #         anomaly_predictions.append(anomaly_prediction_np)
            #         ground_truths.append(ground_truth)
            #
            # anomaly_scores = np.concatenate(anomaly_scores).flatten()
            # ground_truths = np.concatenate(ground_truths).flatten()
            # anomaly_predictions = np.concatenate(anomaly_predictions).flatten()
            # pre = precision_score(ground_truths, anomaly_predictions)
            # rec = recall_score(ground_truths, anomaly_predictions)
            # f1 = f1_score(ground_truths, anomaly_predictions)
            # C = confusion_matrix(ground_truths, anomaly_predictions)



            # with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
            #     for i in range(task_num):
            #         for j in range(setsz):
            #             _, spt_logits = fnet(x_spt[i])
            #             spt_loss = F.mse_loss(spt_logits, x_spt[i])
            #             diffopt.step(spt_loss)
            #
            #         net.eval()
            #         anomaly_predictions = []
            #         ground_truths = []
            #         with torch.no_grad():
            #             for j in range(querysz):
            #                 _, qry_logits = fnet(x_qry[i])
            #                 qry_mse_loss = F.mse_loss(qry_logits, x_qry[i])
            #                 qry_mes = torch.sqrt(torch.pow(qry_logits - x_qry[i], 2.0))
            #                 qry_mes, _ = torch.max(qry_mes, dim=2)
            #                 isNan = qry_mes[0][0].item()
            #                 if isNan != isNan:
            #                     continue
            #                 qry_mes = torch.sigmoid(qry_mes)
            #                 qry_loss = F.mse_loss(qry_mes, y_qry[i]) + qry_mse_loss
            #                 qry_acc = (qry_mes * y_qry[i]).sum().item() / querysz
            #
            #                 length = int(qry_mes.shape[1] * 0.1)
            #                 vals, indices = qry_mes.topk(k=length, dim=1, sorted=True)
            #                 kth_vals_column = vals[:, -1].reshape(-1, 1)
            #                 kth_vals = kth_vals_column.repeat(1, qry_mes.shape[1])
            #                 anomaly_prediction = torch.where(qry_mes > kth_vals, torch.ones_like(qry_mes),
            #                                                  torch.zeros_like(qry_mes))
            #                 anomaly_prediction_np = anomaly_prediction.detach().cpu().numpy()
            #                 ground_truth = y_qry[i].detach().cpu().numpy()
            #                 anomaly_predictions.append(anomaly_prediction_np)
            #                 ground_truths.append(ground_truth)
            #
            #         ground_truths = np.concatenate(ground_truths)
            #         ground_truths = ground_truths.flatten()
            #         anomaly_predictions = np.concatenate(anomaly_predictions).flatten()
            #         pre = precision_score(ground_truths, anomaly_predictions)
            #         rec = recall_score(ground_truths, anomaly_predictions)
            #         f1 = f1_score(ground_truths, anomaly_predictions)
            #         C = confusion_matrix(ground_truths, anomaly_predictions)
            #         if batch_idx % 1 == 0:
            #             print(f'[Epoch {epoch + 1:.2f}] Test Loss: {qry_loss:.2f} | F1: {f1:.2f} | Acc: {qry_acc:.2f}')
            #         log.append({
            #             'epoch': epoch + 1,
            #             'loss': qry_loss,
            #             'acc': qry_acc,
            #             'precision': pre,
            #             'recall': rec,
            #             'f1': f1,
            #             "TP": C[1, 1],
            #             "TN": C[0, 0],
            #             "FP": C[0, 1],
            #             "FN": C[1, 0],
            #             'mode': 'train',
            #             'time': time.time(),
            #         })



def plot(log):
    df = pd.DataFrame(log)
    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'maml-accs.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)


if __name__ == '__main__':
    main()
    # length = int(qry_mes.shape[1] * 0.1)
    # vals, indices = qry_mes.topk(k=length, dim=1, sorted=True)
    # kth_vals_column = vals[:, -1].reshape(-1, 1)
    # kth_vals = kth_vals_column.repeat(1, qry_mes.shape[1])
    # anormaly_prediction = torch.where(qry_mes > kth_vals, torch.ones_like(qry_mes), torch.zeros_like(qry_mes))
