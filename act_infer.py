# 进行act网络的推理
# TODO: 封装O的操作到env

# 使用act网络进行模仿学习
import os
import pickle
import argparse
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import set_seed, load_data_test, compute_dict_mean, detach_dict
from constants import TASK_CONFIG, STATE_DIM
from policy import ACTPolicy

def main(args):
    # TODO: make seed configurable
    set_seed(1)
    # command line parameters
    real_GI = args['real_O']
    save_video = args['save_video']
    ckpt_dir = args['ckpt_dir']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']

    # get task parameters
    task_config = TASK_CONFIG[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len'] # TODO: rm this config
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = STATE_DIM
    # TODO: make backbones configurable
    lr_backbone = 1e-5
    backbone = 'resnet18'

    # ACT parameters
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {
                    'num_queries': args['chunk_size'],
                    'hidden_dim': args['hidden_dim'],
                    'dim_feedforward': args['dim_feedforward'],
                    'backbone': backbone,
                    'enc_layers': enc_layers,
                    'dec_layers': dec_layers,
                    'nheads': nheads,
                    'camera_names': camera_names,
                    }
    config = {
        'ckpt_dir': ckpt_dir,
        'ckpt_name': args['ckpt_name'],
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        # 'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_GI': real_GI,
        'save_video': save_video,
    }

    # test !
    if not real_GI:
        test_on_data(config)
    else:
        test_on_real(config)


def test_on_data(config):
    # TODO: code the envs for testing
    pass

def test_on_real(config):
    # 
    pass
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: screen render
    # just render dx dy + print kb action
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--real_O', action='store', help='infer in GI', required=True)
    parser.add_argument('--save_video', action='store', help='save_video', required=True)
    # for ACT
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    # 这里也进行是否时间集成的消融实验
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))