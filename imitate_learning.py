# 使用act网络进行模仿学习
import os
import pickle
import argparse
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import set_seed, load_data, compute_dict_mean, detach_dict
from constants import TASK_CONFIG, STATE_DIM
from policy import ACTPolicy, CNNMLPPolicy

from config import configs


def main(args):
    # the very first mom to get other parameters
    config_name = args['config_name']
    config = configs[config_name]

    # basic config in config.py
    set_seed(config['seed'])
    # maybe be more gentle
    global device
    device              = config['device']
    ckpt_dir            = config['ckpt_dir']
    policy_class        = config['policy_class']
    task_name           = config['task_name']
    batch_size_train    = config['batch_size']
    batch_size_val      = config['batch_size']
    num_epochs          = config['num_epochs']
    chunk_size          = config['chunk_size']

    # task specific config in constants.py
    task_config         = TASK_CONFIG[task_name]
    dataset_dir         = task_config['dataset_dir']
    num_episodes        = task_config['num_episodes']
    episode_len         = task_config['episode_len']
    camera_names        = task_config['camera_names']
    
    # fixed parameters
    state_dim = STATE_DIM

    # TODO: make these configurable
    # ACT parameters
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {'lr'               : config['lr'],
                     'chunk_size'       : config['chunk_size'],
                     'kl_weight'        : config['kl_weight'],
                     'hidden_dim'       : config['hidden_dim'],
                     'dim_feedforward'  : config['dim_feedforward'],
                     'lr_backbone'      : config['lr_backbone'],
                     'backbone'         : config['backbone'],
                     'enc_layers'       : enc_layers,
                     'dec_layers'       : dec_layers,
                     'nheads'           : nheads,
                     'camera_names'     : task_config['camera_names'],
                     'device'           : device,
                     }
    config = {
        'pretrained'        : config['pretrained'],
        'pretrained_ckpt'   : config['pretrained_ckpt'],
        'num_epochs'        : num_epochs,
        'ckpt_dir'          : ckpt_dir,
        'episode_len'       : episode_len,
        'state_dim'         : state_dim,
        'lr'                : config['lr'],
        'policy_class'      : policy_class,
        'policy_config'     : policy_config,
        'task_name'         : task_name,
        'seed'              : config['seed'],
        'camera_names'      : camera_names,
    }

    train_dataloader, val_dataloader, stats = load_data(dataset_dir, num_episodes, camera_names, chunk_size, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        # 保存这一数据集的mean and std for 推理
        pickle.dump(stats, f)
    
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def train_bc(train_dataloader, val_dataloader, config):
    num_epochs      = config['num_epochs']
    ckpt_dir        = config['ckpt_dir']
    seed            = config['seed']
    policy_config   = config['policy_config']

    set_seed(seed)

    if config['policy_class'] == 'mlp':
        policy = CNNMLPPolicy(policy_config)
    elif config['policy_class'] == 'act':
        policy = ACTPolicy(policy_config)
    
    if config['pretrained']:
        policy.load_can_load(config['pretrained_ckpt'])
    policy = policy.to(device)
    optimizer = policy.configure_optimizers()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)


        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
        if epoch % 20 == 0:
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    # 输出各个data的dtype和shape
    # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
    # print(image_data.shape, qpos_data.shape, action_data.shape, is_pad.shape)
    # exit()
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-c',
                        action='store', type=str, 
                        help=f'which config to be used', 
                        choices=configs.keys(),
                        required=True)
    main(vars(parser.parse_args()))