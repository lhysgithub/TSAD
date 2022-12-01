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
    db = TimeSeriesDatabase(args)
    net = MTAD_GAT(n_features=args.n_features,
                   window_size=args.lookback,
                   out_dim=args.n_features,
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

    meta_opt = optim.Adam(net.parameters(), lr=1e-3)
    with torch.backends.cudnn.flags(enabled=False):
        log = []
        for epoch in range(30):
            # train(db, net, device, meta_opt, epoch, log)
            test(db, net, device, epoch, log)
            # plot(log)


def train(db, net, device, meta_opt, epoch, log):
    net.train()
    n_train_iter = 5
    for batch_idx in range(n_train_iter):
        start_time = time.time()
        x_spt, y_spt, x_qry, y_qry = db.next()
        task_num, setsz, t, d = x_spt.size()
        querysz = x_qry.size(1)
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)
        meta_opt.zero_grad()
        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                for j in range(setsz):
                    _, spt_logits = fnet(x_spt[i])
                    spt_loss = F.mse_loss(spt_logits, x_spt[i])
                    diffopt.step(spt_loss)

                anomaly_predictions = []
                ground_truths = []
                for j in range(querysz):
                    _, qry_logits = fnet(x_qry[i])
                    qry_mse_loss = F.mse_loss(qry_logits, x_qry[i])
                    qry_mes = torch.sqrt(torch.pow(qry_logits - x_qry[i], 2.0))
                    qry_mes, _ = torch.max(qry_mes, dim=2)
                    isNan = qry_mes[0][0].item()
                    if isNan != isNan:
                        continue
                    qry_mes = torch.sigmoid(qry_mes)
                    qry_loss = F.mse_loss(qry_mes, y_qry[i]) + qry_mse_loss
                    qry_acc = (qry_mes * y_qry[i]).sum().item() / querysz
                    qry_loss.backward()
                    meta_opt.step()

                    length = int(qry_mes.shape[1] * 0.1)
                    vals, indices = qry_mes.topk(k=length, dim=1, sorted=True)
                    kth_vals_column = vals[:, -1].reshape(-1, 1)
                    kth_vals = kth_vals_column.repeat(1, qry_mes.shape[1])
                    anomaly_prediction = torch.where(qry_mes > kth_vals, torch.ones_like(qry_mes), torch.zeros_like(qry_mes))
                    anomaly_prediction_np = anomaly_prediction.detach().cpu().numpy()
                    ground_truth = y_qry[i].detach().cpu().numpy()
                    anomaly_predictions.append(anomaly_prediction_np)
                    ground_truths.append(ground_truth)

            ground_truths = np.concatenate(ground_truths)
            ground_truths = ground_truths.flatten()
            anomaly_predictions = np.concatenate(anomaly_predictions).flatten()
            pre = precision_score(ground_truths, anomaly_predictions)
            rec = recall_score(ground_truths, anomaly_predictions)
            f1 = f1_score(ground_truths, anomaly_predictions)
            C = confusion_matrix(ground_truths, anomaly_predictions)
            i = epoch + float(batch_idx) / n_train_iter
            iter_time = time.time() - start_time
            if batch_idx % 1 == 0:
                print(f'[Epoch {i:.2f}] Train Loss: {qry_loss:.2f} | F1: {f1:.2f} | Acc: {qry_acc:.2f} | Time: {iter_time:.2f}')
            log.append({
                'epoch': i,
                'loss': qry_loss,
                'acc': qry_acc,
                'precision': pre,
                'recall': rec,
                'f1': f1,
                "TP": C[1, 1],
                "TN": C[0, 0],
                "FP": C[0, 1],
                "FN": C[1, 0],
                'mode': 'train',
                'time': time.time(),
            })


def test(db, net, device, epoch, log):
    net.train()
    n_test_iter = 1
    for batch_idx in range(n_test_iter):
        x_spt, y_spt, z_spt, x_qry, y_qry, z_qry = db.next('test')
        task_num, spt_inner_batches, spt_inner_batch_size, t, d = x_spt.shape
        qry_inner_batches = x_qry.shape[1]
        inner_opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for i in range(task_num):
            net.train()
            train_losses = []
            for j in range(spt_inner_batches):
                z = torch.from_numpy(z_spt[i][j]).to(device)
                x = torch.from_numpy(x_spt[i][j]).to(device)
                inner_opt.zero_grad()
                preds, spt_logits = net(x)
                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if z.ndim == 3:
                    z = z.squeeze(1)
                spt_loss = torch.sqrt(F.mse_loss(spt_logits, x)) + torch.sqrt(F.mse_loss(preds,z))
                spt_loss.backward()
                inner_opt.step()
                train_losses.append(spt_loss.item())
            print(f"train_epoch: {epoch} loss: {np.array(train_losses).mean()}")

            net.eval()
            pres = []
            recons = []
            with torch.no_grad():
                for j in range(qry_inner_batches):
                    x = torch.from_numpy(x_qry[i][j]).to(device)
                    y = torch.from_numpy(y_qry[i][j]).to(device)
                    z = torch.from_numpy(z_qry[i][j]).to(device)
                    x_hat = torch.cat((x[:, 1:, :], z), dim=1)
                    pre_logits, _ = net(x)
                    _, qry_logits = net(x_hat)
                    recons.append(qry_logits[:, -1, :].detach().cpu().numpy())
                    pres.append(pre_logits.detach().cpu().numpy())
            pres = np.concatenate(pres, axis=0)[:-1]
            recons = np.concatenate(recons, axis=0)[:-1]
            actual = np.reshape(z_qry[i],(-1,38))[:-1]
            # ground_truth_label = db.y_test[100:].flatten()
            ground_truth_label = np.reshape(y_qry[i], (-1, 100))
            ground_truth_label = ground_truth_label[1:][:,0].flatten()
            # ground_truth_label = np.concatenate([prefix,ground_truth_label[-1]])
            # ground_truth_label = ground_truth_label.flatten()
            anomaly_scores = np.zeros_like(actual)
            for i in range(pres.shape[1]):
                a_score = np.sqrt((pres[:, i] - actual[:, i]) ** 2) + np.sqrt((recons[:, i] - actual[:, i]) ** 2)
                anomaly_scores[:, i] = a_score
            anomaly_scores = np.mean(anomaly_scores, 1)

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
            bf_eval = bf_search(anomaly_scores, ground_truth_label, start=0.01, end=2, step_num=100, verbose=False)
            if batch_idx % 1 == 0:
                # print(f'[Epoch {epoch + 1:.2f}] Test Loss: {qry_loss:.2f} | F1: {f1:.2f} | Precision: {pre:.2f} | Recall: {rec:.2f} | Acc: {qry_acc:.2f}')
                print(
                    f'best F1: {bf_eval["f1"]:.2f} | precision: {bf_eval["precision"]:.2f} | Recall: {bf_eval["recall"]:.2f}')

            log.append({
                'epoch': epoch + 1,
                # 'loss': qry_loss,
                # 'acc': qry_acc,
                # 'precision': pre,
                # 'recall': rec,
                # 'f1': f1,
                # "TP": C[1, 1],
                # "TN": C[0, 0],
                # "FP": C[0, 1],
                # "FN": C[1, 0],
                'mode': 'train',
                'time': time.time(),
            })
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
