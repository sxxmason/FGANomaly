#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pickle import load
import os
from torch.utils.data import DataLoader, TensorDataset
import torch as t
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import random
import torch.nn as nn


def seed_all(seed=2020):
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class AdaWeightedLoss(nn.Module):
    def __init__(self, strategy='linear'):
        super(AdaWeightedLoss, self).__init__()
        self.strategy = strategy

    def forward(self, input, target, global_step):
        """
        The reconstruction error will be calculated between x and x', where
        x is a vector of x_dim.

        args:
            input: original values, [bsz,seq,x_dim]
            target: reconstructed values
            global_step: training global step
            strategy: how fast the coefficient w2 shrink to 1.0
        return:
        """
        bsz, seq, x_dim = target.size()

        with t.no_grad():
            # errors: [bsz,seq]
            # w1: [bsz,seq]
            errors = t.sqrt(t.sum((input - target) ** 2, dim=-1))
            error_mean = t.mean(errors, dim=-1)[:, None]
            error_std = t.std(errors, dim=-1)[:, None] + 1e-6
            z_score = (errors - error_mean) / error_std
            neg_z_score = -z_score
            w1 = t.softmax(neg_z_score, dim=-1)

            # exp_z_score: [bsz,seq] -> [bsz,1,seq] -> [bsz,seq,seq] -> [bsz,seq]
            exp_z_score = t.exp(neg_z_score)
            exp_z_score = exp_z_score[:, None, :].repeat(1, seq, 1)
            step_coeff = t.ones(size=(seq, seq), dtype=target.dtype, device=target.device)

            for i in range(seq):
                if self.strategy == 'log':
                    step_coeff[i][i] *= np.log(global_step + np.e - 1)
                elif self.strategy == 'linear':
                    step_coeff[i][i] *= global_step
                elif self.strategy == 'nlog':
                    step_coeff[i][i] *= global_step * np.log(global_step + np.e - 1)
                elif self.strategy == 'quadratic':
                    step_coeff[i][i] *= (global_step ** 2)
                else:
                    raise KeyError('Decay function must be one of [\'log\',\'linear\',\'nlog\',\'quadratic\']')

            exp_z_score = exp_z_score * step_coeff
            w2 = t.sum(exp_z_score, dim=-1) / exp_z_score[:, t.arange(0, seq), t.arange(0, seq)]
            w = w1 * w2
            # normalization
            w = (w / t.sum(w, dim=-1)[:, None])[:, :, None]

        error_matrix = (target - input) ** 2
        return t.sum(error_matrix * w) / (bsz * x_dim)


def normalize(seq):
    return (seq - np.min(seq)) / (np.max(seq) - np.min(seq))


def anomaly_scoring(values, reconstruction_values):
    scores = []
    for v1, v2 in zip(values, reconstruction_values):
        scores.append(np.sqrt(np.sum((v1 - v2) ** 2)))
    return np.array(scores)


def metrics_calculate(values, re_values, labels):
    scores = anomaly_scoring(values, re_values)

    preds, _ = evaluate(labels, scores, adj=False)
    preds_, _ = evaluate(labels, scores, adj=True)

    f1 = f1_score(y_true=labels, y_pred=preds)
    pre = precision_score(y_true=labels, y_pred=preds)
    re = recall_score(y_true=labels, y_pred=preds)

    f1_ = f1_score(y_true=labels, y_pred=preds_)
    pre_ = precision_score(y_true=labels, y_pred=preds_)
    re_ = recall_score(y_true=labels, y_pred=preds_)
    auc = roc_auc_score(y_true=labels, y_score=normalize(scores))

    print('F1 score is [%.5f / %.5f] (before adj / after adj), auc score is %.5f.' % (f1, f1_, auc))
    print('Precision score is [%.5f / %.5f], recall score is [%.5f / %.5f].' % (pre, pre_, re, re_))


def evaluate(labels, scores, step=2000, adj=True):
    # best f1
    min_score = min(scores)
    max_score = max(scores)
    best_f1 = 0.0
    best_preds = None
    for th in tqdm(np.linspace(min_score, max_score, step), ncols=70):
        preds = (scores > th).astype(int)
        if adj:
            preds = adjust_predicts(labels, preds)
        f1 = f1_score(y_true=labels, y_pred=preds)
        if f1 > best_f1:
            best_f1 = f1
            best_preds = preds

    return best_preds, best_f1


def adjust_predicts(label, pred=None):
    predict = pred.astype(bool)
    actual = label > 0.1
    anomaly_state = False
    for i in range(len(label)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
        elif not actual[i]:
            anomaly_state = False

        if anomaly_state:
            predict[i] = True
    return predict.astype(int)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = load(f)
        return data


def get_from_one(ts, window_size, stride):
    ts_length = ts.shape[0]
    samples = []
    for start in np.arange(0, ts_length, stride):
        if start + window_size > ts_length:
            break
        samples.append(ts[start:start+window_size])
    return np.array(samples)


def remove_all_same(train_x, test_x):
    remove_idx = []
    for col in range(train_x.shape[1]):
        if max(train_x[:, col]) == min(train_x[:, col]):
            remove_idx.append(col)
        else:
            train_x[:, col] = normalize(train_x[:, col])

        if max(test_x[:, col]) == min(test_x[:, col]):
            remove_idx.append(col)
        else:
            test_x[:, col] = normalize(test_x[:, col])

    all_idx = set(range(train_x.shape[1]))
    remain_idx = list(all_idx - set(remove_idx))
    return train_x[:, remain_idx], test_x[:, remain_idx]


def load_data(data_prefix, val_size, window_size=120, stride=1, batch_size=64, dataloder=False):
    # root path
    root_path = os.path.join('dataset', data_prefix + '_raw_data')

    # load data from .pkl file
    train_x = load_pickle(os.path.join(root_path, 'train.pkl'))
    test_x = load_pickle(os.path.join(root_path, 'test.pkl'))
    test_y = load_pickle(os.path.join(root_path, 'test_label.pkl')).astype(int)

    # remove columns have 0 variance
    train_x, test_x = remove_all_same(train_x, test_x)

    # train_test_split
    nc = train_x.shape[1]
    train_len = int(len(train_x) * (1-val_size))
    val_x = train_x[train_len:]
    train_x = train_x[:train_len]

    print('Training data:', train_x.shape)
    print('Validation data:', val_x.shape)
    print('Testing data:', test_x.shape)

    if dataloder:
        # windowed data
        train_x = get_from_one(train_x, window_size, stride)
        # train_y has no meaning, only used for TensorDataset
        train_y = np.zeros(len(train_x))

        train_dataset = TensorDataset(t.Tensor(train_x), t.Tensor(train_y))

        data_loader = {"train": DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False),
            "val": val_x,
            "test": (test_x, test_y),
            "nc": nc
        }

        return data_loader
    else:
        return {
            "train": train_x,
            "val": val_x,
            "test": (test_x, test_y),
            "nc": nc
        }
