# copy from act repo shamelessly.

import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed
# 这个交互式后面也没用到啊？？

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        # 这里的norm是在整个task的数据集上计算的
        # 需要保留归一norm以在推理时使用？
        # 嗷，保存了，在ckpt_dir/dataset_stats.pkl
        self.norm_stats = norm_stats
        # IN Yaa, there is not "sim", only record.
        # 所以无需这个变量
        # self.is_sim = None
        # self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            # is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape # batch_size/episode_len, action_dim
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            # Yaa changed status
            # qpos = root['/observations/qpos'][start_ts]
            # qvel = root['/observations/qvel'][start_ts]
            obs_state = root['/obs/state'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/obs/images/{cam_name}'][start_ts]
                
            # get all actions after and including start_ts
            # if is_sim:
            #     action = root['/action'][start_ts:]
            #     action_len = episode_len - start_ts
            # else:
            #     action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
            #     action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
            # IN Yaa, there is not "sim", only record.
            action = root['/action'][max(0, start_ts - 1):] # 先试试-1
            action_len = episode_len - max(0, start_ts - 1)

        # self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        # qpos_data = torch.from_numpy(qpos).float()
        obs_state_data = torch.from_numpy(obs_state).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        # qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        obs_state_data = (obs_state_data - self.norm_stats['obs_state_mean']) / self.norm_stats['obs_state_std']

        return image_data, obs_state_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    # all_qpos_data = []
    all_state_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            # qpos = root['/observations/qpos'][()]
            obs_state = root['/obs/state'][()]
            action = root['/action'][()]
        # all_qpos_data.append(torch.from_numpy(qpos))
        all_state_data.append(torch.from_numpy(obs_state))
        all_action_data.append(torch.from_numpy(action))
    # all_qpos_data = torch.stack(all_qpos_data)
    # 这里的stack是会创建新维度
    # episode_idx, batch_size, state/action_dim
    all_state_data = torch.stack(all_state_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data # ? what is this for?

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    # qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    # qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    # qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
    state_mean = all_state_data.mean(dim=[0, 1], keepdim=True)
    state_std = all_state_data.std(dim=[0, 1], keepdim=True)
    state_std = torch.clip(state_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
            #  "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
                "obs_state_mean": state_mean.numpy().squeeze(), "obs_state_std": state_std.numpy().squeeze(),
            #  "example_qpos": qpos}
            "example_state": obs_state}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    # TODO: make this ratio configurable
    # 居然是按示教轨迹来划分训练集和验证集的
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats


### env utils
# IN Yaa, there is no sim env.
'''
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose
'''
### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
