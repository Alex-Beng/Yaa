# copy from act repo shamelessly.

import os
import pickle
from random import randint
from copy import deepcopy

import win32gui
import win32ui
import win32con

import numpy as np
import torch
import h5py
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from constants import SN
# 增大E键按下后的episode出现的概率

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size,  samp_traj=True):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        # 是否随机裁剪轨迹长度
        # 对于test没用，因为这里是一个image -> actions的映射
        # test需要的仅仅是 images -> actions
        self.samp_traj = samp_traj
        # 这里的norm是在整个task的数据集上计算的
        # 在ckpt_dir/dataset_stats.pkl
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = not self.samp_traj

        # TODO: make this configurable（是否随机选择episode）
        # episode_id = self.episode_ids[index]
        episode_id = np.random.choice(self.episode_ids)
        dataset_path = os.path.join(self.dataset_dir, f'{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            # is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape # episode_len, action_dim
            episode_len, action_dim = original_action_shape

            # 从 /action 寻找第一个 E 不为 0 的位置
            # TODO: add more keys of interest, make it a function?
            e_begin = 0
            for i in range(episode_len):
                if root['/action'][i, SN['E']] != 0:
                    e_begin = i
                    break

            if sample_full_episode:
                start_ts = 0
            else:
                # TODO: make other keys more possible too
                # TODO: make probability configurable
                # 有概率直接短路到E，即开始战斗
                if np.random.rand() < 0.1:
                    start_ts = randint(e_begin, episode_len - 1)
                else:
                    start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            # Yaa changed status
            obs_state = root['/obs/state'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/obs/images/{cam_name}'][start_ts]
            
            # Hack trick, 实际机器人/窗口响应需要时间
            # TODO: make this configurable
            ts_offset = -1
            
            # 只要chunk size内的action
            s_idx = max(0, start_ts + ts_offset)
            e_idx = min(start_ts + self.chunk_size + ts_offset, episode_len) # need offset too
            action = root['/action'][s_idx:e_idx]
            action_len = e_idx - s_idx
        
        padded_action = np.zeros((self.chunk_size, action_dim), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size, dtype=np.bool)
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

        # channel last -> channel first
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        # 图像后面还会在policy中基于ImageNet数据集的均值和方差进行归一化
        image_data = image_data / 255.0

        # 只对action的最后两个维度进行norm
        mouse_action = action_data[:, -2:]
        # TODO: 设置两种norm方式
        if True:
            mouse_action = (mouse_action - self.norm_stats["mouse_action_min"]) / (self.norm_stats["mouse_action_max"] - self.norm_stats["mouse_action_min"])
            # print(f'min mouse action: {torch.min(mouse_action, dim=0)}, max mouse action: {torch.max(mouse_action, dim=0)}')
        else:
            mouse_action = (mouse_action - self.norm_stats["mouse_action_mean"]) / self.norm_stats["mouse_action_std"]
            # print(f'min mouse action: {torch.min(mouse_action, dim=0)}, max mouse action: {torch.max(mouse_action, dim=0)}')
         
        action_data[:, -2:] = mouse_action
        # 设置pad后的action为0
        action_data[action_len:, :] = 0

        return image_data, obs_state_data, action_data, is_pad

# 用于test的数据集
# one traj in.
# return image, action in each ts
class EpisodicDatasetTest(torch.utils.data.Dataset):
    def __init__(self, episode_id, dataset_dir, camera_names, norm_stats, samp_traj=False):
        super(EpisodicDatasetTest).__init__()
        # only id !
        self.episode_id = episode_id
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.samp_traj = samp_traj
        
        f = h5py.File(os.path.join(self.dataset_dir, f'{self.episode_id}.hdf5'), 'r')
        self.episode_len = f['/action'].shape[0]
        self.hdf5_handle = f
        
        if self.samp_traj:
            self.start_ts = np.random.choice(self.episode_len)
            self.episode_len = self.episode_len - self.start_ts
        else:
            self.start_ts = 0
        
        preprocess = lambda action: (action - self.norm_stats['action_mean']) / self.norm_stats['action_std']
        postprocess = lambda action: (action * self.norm_stats['action_std']) + self.norm_stats['action_mean']
        self.preprocess = preprocess
        self.postprocess = postprocess
    def __len__(self):
        return self.episode_len
    
    def __getitem__(self, index):
        ts = self.start_ts + index
        
        # image, action
        # 不要做预处理，只要跟原始捕获一致即可
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = self.hdf5_handle[f'/obs/images/{cam_name}'][ts]
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        # stack 多了一个维度，k h w c 
        all_cam_images = np.stack(all_cam_images, axis=0)
    
        image_data = torch.from_numpy(all_cam_images)
        # image_data = torch.einsum('k h w c -> k c h w', image_data)
        # image_data = image_data / 255.0
        
        action = self.hdf5_handle['/action'][ts]
        # action = self.preprocess(action)

        return image_data, action

def get_norm_stats(dataset_dir, num_episodes):
    # all_qpos_data = []
    all_state_data = []
    all_action_data = []
    max_episode_len = 0
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            obs_state = root['/obs/state'][()]
            action = root['/action'][()]
        
        all_state_data.append(torch.from_numpy(obs_state))
        all_action_data.append(torch.from_numpy(action))
        max_episode_len = max(max_episode_len, action.shape[0])

    # concat 所有 episode 的 state/action 到 [\sum_i episode_len, state/action_dim]
    all_state_data = torch.cat(all_state_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # keyboard state and action 不需要进行norm，全是0/1
    # mouse state 不需要进行norm，全是0 
    mouse_action = all_action_data[:, -2:]

    # 归一化，减去最小值，除以最大值
    mouse_action_min = torch.min(mouse_action, dim=0).values
    mouse_action_max = torch.max(mouse_action, dim=0).values
    
    # 标准化，减去均值，除以标准差
    mouse_action_mean = torch.mean(mouse_action, dim=0)
    mouse_action_std = torch.std(mouse_action, dim=0)


    stats = {
        "mouse_action_min": mouse_action_min.numpy().squeeze(), 
        "mouse_action_max": mouse_action_max.numpy().squeeze(),
        "mouse_action_mean": mouse_action_mean.numpy().squeeze(), 
        "mouse_action_std": mouse_action_std.numpy().squeeze(),
        "max_episode_len": max_episode_len
    }
    return stats, max_episode_len


def load_data(dataset_dir, num_episodes, camera_names, chunk_size, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    # TODO: make this ratio configurable
    # 居然是按示教轨迹来划分训练集和验证集的
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    # TODO: make this configurable
    train_indices = deepcopy(shuffled_indices)
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats & max_episode for qpos and action
    # use max_episode to pad in dataloader
    norm_stats, max_episode_len = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, chunk_size, True)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, chunk_size, True)
    # print(f'batch size train: {batch_size_train}, batch size val: {batch_size_val}')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, 
                                  shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1,)
                                #   collate_fn=MyCollate(max_episode_len).collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val,
                                shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1,)
                                # collate_fn=MyCollate(max_episode_len).collate_fn)

    return train_dataloader, val_dataloader, norm_stats

def load_data_test(dataset_dir, ckpt_dir, episode_id, camera_names):
    # 不进行训练集和验证集的划分
    # norm_stats 从pkl文件中读取，训练时必然保存的
    norm_stats = None
    with open(os.path.join(ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        norm_stats = pickle.load(f)
    if norm_stats is None:
        raise ValueError('Cannot load dataset stats from dataset_stats.pkl')

    test_dataset = EpisodicDatasetTest(episode_id, dataset_dir, camera_names, norm_stats)

    return test_dataset, norm_stats

    

class MyCollate:
    def __init__(self, max_episode_len):
        self.max_len = max_episode_len

    def collate_fn(self, batch):
        # batch is [(image, state, action, is_pad), ...]
        # image: [num_cams, C, H, W]
        # state: [state_dim]
        # action: [episode_len, action_dim]
        # is_pad: [episode_len]
        image, state, action, is_pad = zip(*batch)

        # pad action and is_pad
        action = pad_sequence(action, batch_first=True, padding_value=0)
        is_pad = pad_sequence(is_pad, batch_first=True, padding_value=1)

        return torch.stack(image), torch.stack(state), action, is_pad


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

# windows capture
def capture(hwnd):
    # 获取窗口的设备上下文
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    # 获取窗口的大小
    left, top, right, bot = win32gui.GetClientRect(hwnd)
    width = right - left
    height = bot - top

    # 创建一个位图来存储捕获内容
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    # 使用BitBlt捕获窗口图像
    result = saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

    # 将捕获的图像转换为numpy数组
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    imdata = np.frombuffer(bmpstr, dtype='uint8')
    imdata.shape = (height, width, 4)

    # 清理资源
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    # 返回numpy数组
    return imdata



if __name__ == "__main__":
    dts = EpisodicDatasetTest([0])