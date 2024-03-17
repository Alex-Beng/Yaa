# 进行act网络的推理
# TODO: 封装O的操作到env

# 使用act网络进行模仿学习
import os
import time
import pickle
import argparse
from copy import deepcopy

import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange

from utils import set_seed, load_data_test, compute_dict_mean, detach_dict
from constants import TASK_CONFIG, STATE_DIM, SN_idx2key
from policy import ACTPolicy, CNNMLPPolicy
from gi_env import GIDataEnv, GIRealEnv
from config import infer_configs

import IPython
e = IPython.embed

# TODO: make device configurable
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

def main(args):
    # the very first mom to get other parameters
    config_name = args['config_name']
    config = infer_configs[config_name]
    
    # basic config in config.py
    set_seed(config['seed'])
    global device   
    device              = config['device']
    ckpt_dir            = config['ckpt_dir']
    policy_class        = config['policy_class']
    task_name           = config['task_name']
    chunk_size          = config['chunk_size']
    backbone            = config['backbone']
    
    ckpt_name           = args['ckpt_name']
    real_GI             = args['real_O']
    save_video          = args['save_video']
    onscreen_render     = args['onscreen_render']
    temporal_agg        = args['temporal_agg']

    # get task parameters
    task_config         = TASK_CONFIG[task_name]
    dataset_dir         = task_config['dataset_dir']
    num_episodes        = task_config['num_episodes']
    episode_len         = task_config['episode_len'] # TODO: rm this config
    camera_names        = task_config['camera_names']

    # fixed parameters
    state_dim = STATE_DIM

    # TODO: make these configurable
    # ACT parameters
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {
                    'chunk_size'        : chunk_size,
                    'hidden_dim'        : config['hidden_dim'],
                    'dim_feedforward'   : config['dim_feedforward'],
                    'backbone'          : backbone,
                    'enc_layers'        : enc_layers,
                    'dec_layers'        : dec_layers,
                    'nheads'            : nheads,
                    'camera_names'      : camera_names,
                    'device'            : device,
                    }
    config = {
        'ckpt_dir'          : ckpt_dir,
        'ckpt_name'         : ckpt_name,
        'num_episodes'      : num_episodes,
        'episode_len'       : episode_len,
        'state_dim'         : state_dim,
        'policy_class'      : policy_class,
        'onscreen_render'   : onscreen_render,
        'policy_config'     : policy_config,    

        'task_name'         : task_name,
        'seed'              : config['seed'],
        'temporal_agg'      : chunk_size,
        'camera_names'      : camera_names,
        'real_GI'           : real_GI,
        'save_video'        : save_video,
        'video_dir'         : config['video_dir'],
    }

    if save_video and not os.path.exists(config['video_dir']):
        os.makedirs(config['video_dir'])

    # test !
    if not real_GI:
        test_on_data(config)
    else:
        test_on_real(config)


def test_on_data(config):
    set_seed(config['seed'])
    ckpt_dir        = config['ckpt_dir']
    ckpt_name       = config['ckpt_name']
    state_dim       = config['state_dim']
    onscreen_render = config['onscreen_render']
    policy_config   = config['policy_config']
    camera_names    = config['camera_names']
    task_name       = config['task_name']
    max_timestamps  = config['episode_len']
    max_episodes    = config['num_episodes']
    temporal_agg    = config['temporal_agg']
    save_video      = config['save_video']

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if config['policy_class'] == 'mlp':
        policy = CNNMLPPolicy(policy_config)
    elif config['policy_class'] == 'act':
        policy = ACTPolicy(policy_config)
    
    loading_status =policy.load_state_dict(torch.load(ckpt_path))
    print(f'loading status: {loading_status}')
    policy = policy.to(device)
    policy.eval()
    print(f'policy loaded from {ckpt_path}')

    # load dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    pre_process = lambda state: (state - stats['obs_state_mean']) / stats['obs_state_std']
    post_process = lambda action: action * stats['action_std'] + stats['action_mean']
    print(f'stats: {stats}')
    
    # load env
    env = GIDataEnv(config)

    # TODO: make query freq 独立于 chunk size，使得在使用时间集成时可以延迟几个ts
    # TODO: 研究独立后的query freq对于成功率的影响
    query_frequecy = policy_config['chunk_size']
    if temporal_agg:
        query_frequecy = 1
        chunk_size = policy_config['chunk_size'] # TODO: explain this
    
    # TODO: make this scale configurable
    max_timestamps = int(max_timestamps * 1)

    for episode_id in range(max_episodes):
        print(f'episode {episode_id}')
        # reset env
        env.reset()

        if onscreen_render:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        
        if temporal_agg:
            # max_ts, max_ts+chunk size, state_dim
            # 记录推理的actions ?
            all_time_actions = torch.zeros([max_timestamps, max_timestamps+chunk_size, state_dim])
            all_time_actions = all_time_actions.to(device)
        # IN Yaa, the state is the mskb
        # 同时引入假设，初始为 [0] * state_dim
        # 所以需要在推理的时候，维护一个 state 
        state_history = torch.zeros([1, max_timestamps, state_dim])
        state_history = state_history.to(device)

        # mksb state
        # 最后三个为鼠标滚轮, dx, dy。无需考虑状态
        curr_state = np.zeros([state_dim]) # easier for cpu operations
        image_list = [] # for video?
        state_list = []
        target_state_list = []
        if save_video:
            video_path = os.path.join(config['video_dir'], f'{task_name}_{ckpt_name}_{episode_id}.mp4')
            # TODO: fix the hardcode frame size
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))

        with torch.inference_mode():
            for t in range(max_timestamps):
                obs = env.observation()
                # print(curr_state)
                # state 需要 preprocess
                state_numpy = deepcopy(curr_state)
                state = pre_process(state_numpy)
                # make it to [1, state_dim]
                state = torch.from_numpy(state).float().unsqueeze(0)
                state = state.to(device)
                state_history[0, t] = state
                
                # get image from obs
                image_dict, ground_action = obs
                # print(image_dict.keys())
                curr_image = image_preprocess(image_dict, camera_names)
                # torch.Size([1, 2, 3, 480, 640])
                # print(curr_image.shape) 

                # feed image & state to policy
                if t % query_frequecy == 0:
                    # predict 一个 action chunk
                    # 1, chunk size, state_dim
                    t0 = time.time()
                    print(np.round(state_numpy, 2))
                    all_actions = policy(state, curr_image)
                    print(f'policy cost time: {time.time() - t0}')
                
                # 进行时间集成！
                if temporal_agg:
                    all_time_actions[[t], t:t+chunk_size] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    # TODO: make it configurable
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
                    exp_weights = exp_weights.to(device)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    # 就是每隔 chunk size推理一次
                    raw_action = all_actions[:, t % query_frequecy]
                # 保留sigmoid之前的 mro, dx, dy
                
                # 需要后处理 action
                # sigmoid 一下
                # raw_action = torch.sigmoid(raw_action)
                raw_action = raw_action.squeeze(0).cpu().numpy()
                print(np.round(raw_action, 2))
                # 恢复到采集时候的分布
                action = post_process(raw_action)
                # print action in 两位小数
                print(np.round(action, 1))
                # print(ground_action)

                # 需要把action到离散状态
                # TODO: make threshold configurable
                # 把 action 中 < min_thre 的部分置为 0, > max_thre 的部分置为 1
                min_thre = 0.5
                max_thre = 0.5
                action_bin = np.zeros_like(action, dtype=np.int8)
                action_bin[action < min_thre] = 0
                action_bin[action >= max_thre] = 1

                target_state = action
                # print(np.max(action), np.min(action))
                
                # action影响curr_state
                # 获得发生变动的键盘状态，进而获得实际 人的动作
                human_actions = []
                human_actions_gt = []
                show_updown = False
                for state_id in range(state_dim-3):
                    if show_updown:
                        if action_bin[state_id] == curr_state[state_id]:
                            continue
                        else:
                            human_action = f"{SN_idx2key[state_id]} {'up' if action_bin[state_id] == 0 else 'down'}"
                            if human_action[0] == ' ':
                                human_action = 'sp' + human_action[1:]
                            # print(f'append in {state_id}')
                            human_actions.append(human_action)
                    else:
                        if action_bin[state_id]:
                            human_action = f"{SN_idx2key[state_id]}"
                            human_actions.append(human_action)
                        if ground_action[state_id]:
                            human_action = f"{SN_idx2key[state_id]}"
                            human_actions_gt.append(human_action)
                    curr_state[state_id] = action_bin[state_id]
                dx, dy = action[-2], action[-1]
                dx_gt, dy_gt = ground_action[-2], ground_action[-1]
                print(human_actions)
                print(human_actions_gt)
                
                # do nothing in data env actually
                if not env.step(action):
                    break

                # for visualization
                state_list.append(state_numpy)
                target_state_list.append(target_state)
                
                # update curr frame
                # TODO: plot dx dy in frame
                
                image_dict = env.render()
                # 先把图像拼接起来
                curr_image = np.concatenate([image_dict[cam_name] for cam_name in camera_names], axis=1)
                # frame 上显示 str
                kb_events_str = ','.join(human_actions)
                kb_events_str_gt = ','.join(human_actions_gt)
                curr_image = cv2.putText(curr_image, kb_events_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
                curr_image = cv2.putText(curr_image, kb_events_str_gt, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)

                # 在frame 中间绘制一个箭头，表示dx dy
                dx, dy = int(dx), int(dy)
                dx_gt, dy_gt = int(dx_gt), int(dy_gt)
                curr_image = cv2.arrowedLine(curr_image, (640, 240), (640+dx, 240+dy), (0, 0, 255), 2)
                # 在旁边绘制另一个 predicted dx dy
                curr_image = cv2.arrowedLine(curr_image, (320, 240), (320+dx, 240+dy), (0, 0, 255), 2)

                curr_image = cv2.arrowedLine(curr_image, (640, 240), (640+dx_gt, 240+dy_gt), (0, 255, 0), 2)
                if onscreen_render:
                    cv2.imshow('image', curr_image)
                    cv2.waitKey(1)
                if save_video:
                    out.write(curr_image)                    
                print(f'step {t}/{max_timestamps}')
    


def test_on_real(config):
    # 
    pass

def image_preprocess(image_dict, camera_names):
    # 图像进入推理前的预处理
    # srds，在policy中还有一个normalize to ImageNet分布的操作
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(image_dict[cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().unsqueeze(0)
    curr_image = curr_image.to(device)
    return curr_image


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-c',
                        action='store', type=str, 
                        help='which config to be used', 
                        choices=infer_configs.keys(),
                        required=True)
    parser.add_argument('--ckpt_name',
                        action='store', type=str,
                        help='which ckpt to infer', 
                        default='policy_best.ckpt',
                        required=False)
    parser.add_argument('--real_O', 
                        action='store_true', 
                        help='infer in GI')
    parser.add_argument('--save_video', 
                        action='store_true', 
                        help='save_video',
                        default=True)
    parser.add_argument('--temporal_agg', 
                        action='store_true',
                        help='temporal_agg',
                        default=True)
    parser.add_argument('--onscreen_render',
                        action='store_true',
                        help='onscreen_render',
                        default=True)
    
    main(vars(parser.parse_args()))