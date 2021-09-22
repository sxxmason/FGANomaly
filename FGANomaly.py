#!/usr/bin/env python
# -*- coding: utf-8 -*-
import h5py
import os
from time import time
from copy import deepcopy

import torch as t
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
from torch.nn import MSELoss, BCELoss
import numpy as np
from sklearn.metrics import mean_squared_error

from utils import seed_all, metrics_calculate, AdaWeightedLoss


seed_all(2021)


class RNNEncoder(nn.Module):
    """
    An implementation of Encoder based on Recurrent neural networks.
    """
    def __init__(self, inp_dim, z_dim, hidden_dim, rnn_hidden_dim, num_layers, bidirectional=False, cell='lstm'):
        """
        args:
            inp_dim: dimension of input value
            z_dim: dimension of latent code
            hidden_dim: dimension of fully connection layers
            rnn_hidden_dim: dimension of rnn cell hidden states
            num_layers: number of layers of rnn cell
            bidirectional: whether use BiRNN cell
            cell: one of ['lstm', 'gru', 'rnn']
        """
        super(RNNEncoder, self).__init__()

        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.linear1 = nn.Linear(inp_dim, hidden_dim)
        if bidirectional:
            self.linear2 = nn.Linear(self.rnn_hidden_dim * 2, z_dim)
        else:
            self.linear2 = nn.Linear(self.rnn_hidden_dim, z_dim)

        if cell == 'lstm':
            self.rnn = nn.LSTM(hidden_dim,
                               rnn_hidden_dim,
                               num_layers=num_layers,
                               bidirectional=bidirectional)
        elif cell == 'gru':
            self.rnn = nn.GRU(hidden_dim,
                              rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(hidden_dim,
                              rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional)

    def forward(self, inp):
        # inp shape: [bsz, seq_len, inp_dim]
        self.rnn.flatten_parameters()
        inp = inp.permute(1, 0, 2)
        rnn_inp = t.tanh(self.linear1(inp))
        rnn_out, _ = self.rnn(rnn_inp)
        z = self.linear2(rnn_out).permute(1, 0, 2)
        return z


class RNNDecoder(nn.Module):
    """
    An implementation of Decoder based on Recurrent neural networks.
    """
    def __init__(self, inp_dim, z_dim, hidden_dim, rnn_hidden_dim, num_layers, bidirectional=False, cell='lstm'):
        """
        args:
            Reference argument annotations of RNNEncoder.
        """
        super(RNNDecoder, self).__init__()

        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.linear1 = nn.Linear(z_dim, hidden_dim)
        if bidirectional:
            self.linear2 = nn.Linear(self.rnn_hidden_dim * 2, inp_dim)
        else:
            self.linear2 = nn.Linear(self.rnn_hidden_dim, inp_dim)

        if cell == 'lstm':
            self.rnn = nn.LSTM(hidden_dim,
                               rnn_hidden_dim,
                               num_layers=num_layers,
                               bidirectional=bidirectional)
        elif cell == 'gru':
            self.rnn = nn.GRU(hidden_dim,
                              rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(hidden_dim,
                              rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional)

    def forward(self, z):
        # z shape: [bsz, seq_len, z_dim]
        self.rnn.flatten_parameters()
        z = z.permute(1, 0, 2)
        rnn_inp = t.tanh(self.linear1(z))
        rnn_out, _ = self.rnn(rnn_inp)
        re_x = self.linear2(rnn_out).permute(1, 0, 2)
        return re_x


class RNNAutoEncoder(nn.Module):
    def __init__(self, inp_dim, z_dim, hidden_dim, rnn_hidden_dim, num_layers, bidirectional=False, cell='lstm'):

        super(RNNAutoEncoder, self).__init__()

        self.encoder = RNNEncoder(inp_dim, z_dim, hidden_dim, rnn_hidden_dim,
                                  num_layers, bidirectional=bidirectional, cell=cell)
        self.decoder = RNNDecoder(inp_dim, z_dim, hidden_dim, rnn_hidden_dim,
                                  num_layers, bidirectional=bidirectional, cell=cell)

    def forward(self, inp):
        # inp shape: [bsz, seq_len, inp_dim]
        z = self.encoder(inp)
        re_inp = self.decoder(z)
        return re_inp, z


class MLPDiscriminator(nn.Module):
    def __init__(self, inp_dim, hidden_dim):
        super(MLPDiscriminator, self).__init__()

        self.dis = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        seq, df = inp.shape
        c = self.dis(inp)
        return c.view(seq)


class FGANomalyModel(object):
    def __init__(self, ae, dis_ar, data_loader, **kwargs):
        self.params = kwargs
        self.print_param()
        self.print_model(ae, dis_ar)

        self.device = kwargs['device']
        self.lr = kwargs['lr']
        self.epoch = kwargs['epoch']
        self.window_size = kwargs['window_size']
        self.early_stop = kwargs['early_stop']
        self.early_stop_tol = kwargs['early_stop_tol']
        self.if_scheduler = kwargs['if_scheduler']

        self.adv_rate = kwargs['adv_rate']
        self.dis_ar_iter = kwargs['dis_ar_iter']

        self.weighted_loss = kwargs['weighted_loss']
        self.strategy = kwargs['strategy']

        self.ae = ae.to(self.device)
        self.dis_ar = dis_ar.to(self.device)
        self.data_loader = data_loader

        self.mse = MSELoss()
        self.bce = BCELoss()
        self.ada_mse = AdaWeightedLoss(self.strategy)

        self.ae_optimizer = Adam(params=self.ae.parameters(), lr=self.lr)
        self.ae_scheduler = lr_scheduler.StepLR(optimizer=self.ae_optimizer,
                                                step_size=kwargs['scheduler_step_size'],
                                                gamma=kwargs['scheduler_gamma'])
        self.ar_optimizer = Adam(params=self.dis_ar.parameters(), lr=self.lr)
        self.ar_scheduler = lr_scheduler.StepLR(optimizer=self.ar_optimizer,
                                                step_size=kwargs['scheduler_step_size'],
                                                gamma=kwargs['scheduler_gamma'])

        self.cur_step = 0
        self.cur_epoch = 0
        self.best_ae = None
        self.best_dis_ar = None
        self.best_val_loss = np.inf
        self.val_loss = None
        self.early_stop_count = 0
        self.re_loss = None
        self.adv_dis_loss = None
        self.time_per_epoch = None

    def train(self):
        print('*' * 20 + 'Start training' + '*' * 20)
        for i in range(self.epoch):
            self.cur_epoch += 1
            self.train_epoch()
            self.validate()

            if self.val_loss < self.best_val_loss and self.best_val_loss - self.val_loss >= 1e-4:
                self.best_val_loss = self.val_loss
                self.best_ae = deepcopy(self.ae)
                self.best_dis_ar = deepcopy(self.dis_ar)
                self.save_best_model()
                self.early_stop_count = 0
            elif self.early_stop:
                self.early_stop_count += 1
                if self.early_stop_count > self.early_stop_tol:
                    print('*' * 20 + 'Early stop' + '*' * 20)
                    return
            else:
                pass

            print('[Epoch %d/%d] current training loss is %.5f, val loss is %.5f, adv loss is %.5f, '
                  'time per epoch is %.5f' % (i+1, self.epoch, self.re_loss, self.val_loss,
                                              self.adv_dis_loss, self.time_per_epoch))

    def train_epoch(self):
        start_time = time()
        for x, _ in self.data_loader['train']:
            self.cur_step += 1
            x = x.to(self.device)

            for _ in range(self.dis_ar_iter):
                self.dis_ar_train(x)
            self.ae_train(x)
        end_time = time()
        self.time_per_epoch = end_time - start_time
        if self.if_scheduler:
            self.ar_scheduler.step()
            self.ae_scheduler.step()

    def dis_ar_train(self, x):
        self.ar_optimizer.zero_grad()

        re_x, z = self.ae(x)
        soft_label, hard_label = self.value_to_label(x, re_x)

        actual_normal = x[t.where(hard_label == 0)]
        re_normal = re_x[t.where(hard_label == 0)]
        actual_target = t.ones(size=(actual_normal.shape[0],), dtype=t.float, device=self.device)
        re_target = t.zeros(size=(actual_normal.shape[0],), dtype=t.float, device=self.device)

        re_logits = self.dis_ar(re_normal)
        actual_logits = self.dis_ar(actual_normal)

        re_dis_loss = self.bce(input=re_logits, target=re_target)
        actual_dis_loss = self.bce(input=actual_logits, target=actual_target)

        dis_loss = re_dis_loss + actual_dis_loss
        dis_loss.backward()
        self.ar_optimizer.step()

    def dis_ar_train_no_filter(self, x):
        self.ar_optimizer.zero_grad()

        bsz, seq, fd = x.shape
        re_x, z = self.ae(x)

        re_x = re_x.contiguous().view(bsz * seq, fd)
        x = x.contiguous().view(bsz * seq, fd)

        actual_target = t.ones(size=(x.shape[0],), dtype=t.float, device=self.device)
        re_target = t.zeros(size=(re_x.shape[0],), dtype=t.float, device=self.device)

        re_logits = self.dis_ar(re_x)
        actual_logits = self.dis_ar(x)

        re_dis_loss = self.bce(input=re_logits, target=re_target)
        actual_dis_loss = self.bce(input=actual_logits, target=actual_target)

        dis_loss = re_dis_loss + actual_dis_loss
        dis_loss.backward()
        self.ar_optimizer.step()

    def ae_train(self, x):
        bsz, seq, fd = x.shape
        self.ae_optimizer.zero_grad()

        re_x, z = self.ae(x)

        # reconstruction loss
        if self.weighted_loss:
            self.re_loss = self.ada_mse(re_x, x, self.cur_step)
        else:
            self.re_loss = self.mse(re_x, x)

        # adversarial loss
        ar_inp = re_x.contiguous().view(bsz*seq, fd)
        actual_target = t.ones(size=(ar_inp.shape[0],), dtype=t.float, device=self.device)
        re_logits = self.dis_ar(ar_inp)
        self.adv_dis_loss = self.bce(input=re_logits, target=actual_target)

        loss = self.re_loss + self.adv_dis_loss * self.adv_rate
        loss.backward()
        self.ae_optimizer.step()

    def validate(self):
        self.ae.eval()
        re_values = self.value_reconstruction_val(self.data_loader['val'], self.window_size)
        self.val_loss = mean_squared_error(y_true=self.data_loader['val'][:len(re_values)], y_pred=re_values)
        self.ae.train()

    def test(self, load_from_file=False):
        if load_from_file:
            self.load_best_model()

        self.best_ae.eval()

        test_x, test_y = self.data_loader['test']
        re_values = self.value_reconstruction_val(test_x, self.window_size, val=False)

        values = test_x[:len(re_values)]
        labels = test_y[:len(re_values)]
        metrics_calculate(values, re_values, labels)
        self.save_result(values, re_values, labels)

    def value_reconstruction_val(self, values, window_size, val=True):
        piece_num = len(values) // window_size
        reconstructed_values = []
        for i in range(piece_num):
            raw_values = values[i * window_size:(i + 1) * window_size, :]
            raw_values = t.tensor([raw_values], dtype=t.float).to(self.device)
            if val:
                reconstructed_value_, z = self.ae(raw_values)
            else:
                reconstructed_value_, z = self.best_ae(raw_values)

            reconstructed_value_ = reconstructed_value_.squeeze().detach().cpu().tolist()
            reconstructed_values.extend(reconstructed_value_)
        return np.array(reconstructed_values)

    def value_to_label(self, values, re_values):
        with t.no_grad():
            errors = t.sqrt(t.sum((values - re_values) ** 2, dim=-1))
            error_mean = t.mean(errors, dim=-1)[:, None]
            error_std = t.std(errors, dim=-1)[:, None] + 1e-6
            z_score = (errors - error_mean) / error_std
            z_score = z_score * (1 - 1 / self.cur_epoch)

            soft_label = t.sigmoid(z_score)
            rand = t.rand_like(soft_label)
            hard_label = (soft_label > rand).float()
            return soft_label, hard_label

    def save_best_model(self):
        if not os.path.exists(self.params['best_model_path']):
            os.makedirs(self.params['best_model_path'])

        t.save(self.best_ae, os.path.join(self.params['best_model_path'],
                                          'ae_'+str(self.params['strategy'])+'_'+str(self.params['adv_rate'])+'.pth'))
        t.save(self.best_dis_ar, os.path.join(self.params['best_model_path'],
                                              'dis_'+str(self.params['strategy'])+'_'+str(self.params['adv_rate'])+'.pth'))

    def load_best_model(self):
        self.best_ae = t.load(os.path.join(self.params['best_model_path'], 'ae.pth'))
        self.best_dis_ar = t.load(os.path.join(self.params['best_model_path'], 'dis_ar.pth'))

    def save_result(self, values, re_values, labels):
        if not os.path.exists(self.params['result_path']):
            os.makedirs(self.params['result_path'])

        with h5py.File(os.path.join(self.params['result_path'], 'result_'+str(self.params['strategy'])+'_'+str(self.params['adv_rate'])+'.h5'), 'w') as f:
            f['values'] = values
            f['re_values'] = re_values
            f['labels'] = labels

    def print_param(self):
        print('*'*20+'parameters'+'*'*20)
        for k, v in self.params.items():
            print(k+' = '+str(v))
        print('*' * 20 + 'parameters' + '*' * 20)

    def print_model(self, ae, dis_ar):
        print(ae)
        print(dis_ar)