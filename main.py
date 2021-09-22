#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch as t
from utils import load_data, seed_all
from FGANomaly import FGANomalyModel, RNNAutoEncoder, MLPDiscriminator
import os


seed_all(2021)


params = {
    'data_prefix': 'msl',
    'val_size': 0.3,
    'batch_size': 256,
    'stride': 1,
    'window_size': 120,

    'z_dim': 10,
    'hidden_dim': 50,  # 50 for msl, smap and swat, 100 for wadi
    'rnn_hidden_dim': 50,  # 50 for msl, smap and swat, 100 for wadi
    'num_layers': 1,
    'bidirectional': True,
    'cell': 'lstm',  # 'lstm' for msl, smap and swat, 'gru' for wadi

    'device': t.device('cuda:3' if t.cuda.is_available() else 'cpu'),
    'lr': 3e-4,
    'if_scheduler': True,  # whether use lr scheduler
    'scheduler_step_size': 5,
    'scheduler_gamma': 0.5,

    'epoch': 100,
    'early_stop': True,
    'early_stop_tol': 10,

    'weighted_loss': True,
    'strategy': 'linear',

    'adv_rate': 0.01,
    'dis_ar_iter': 1,

    'best_model_path': os.path.join('rnn_output', 'best_model'),
    'result_path': os.path.join('rnn_output'),
}


def main():
    data = load_data(data_prefix=params['data_prefix'],
                     val_size=params['val_size'],
                     window_size=params['window_size'],
                     stride=params['stride'],
                     batch_size=params['batch_size'],
                     dataloder=True)

    model = FGANomalyModel(ae=RNNAutoEncoder(inp_dim=data['nc'],
                                             z_dim=params['z_dim'],
                                             hidden_dim=params['hidden_dim'],
                                             rnn_hidden_dim=params['rnn_hidden_dim'],
                                             num_layers=params['num_layers'],
                                             bidirectional=params['bidirectional'],
                                             cell=params['cell']),
                           dis_ar=MLPDiscriminator(inp_dim=data['nc'],
                                                   hidden_dim=params['hidden_dim']),
                           data_loader=data, **params)
    model.train()
    model.test()


if __name__ == '__main__':
    for ss in ['log', 'linear', 'nlog', 'quadratic']:
        params['strategy'] = ss
        for ar in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            params['adv_rate'] = ar
            main()
