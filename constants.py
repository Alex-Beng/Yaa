# 储存常量

DT = 0.05

# state name -> dim
from detr.util.constants import SN_idx2key, STATE_DIM

# key of interest -> idx
SN = dict(zip(SN_idx2key, range(STATE_DIM)))

# TODO: is there any useful lib to do this?
key2scancode = {
    'W': 0x11,
    'A': 0x1E,
    'S': 0x1F,
    'D': 0x20,
    'Q': 0x10,
    'E': 0x12,
    'X': 0x2D,
    ' ': 0x39,
    'LS': 0x1D,
    'T': 0x14,
    'Z': 0x2C,
    '1': 0x02,
    '2': 0x03,
    '3': 0x04,
    '4': 0x05,
}

# generate from copilot
# the up scancode of the key
# 等抄了GIA的postmessage再说吧，目前没有用
key2scancode_ups = {
    'W': 0x91,
    'A': 0x9E,
    'S': 0x9F,
    'D': 0xA0,
    'Q': 0x90,
    'E': 0x92,
    'X': 0xAD,
    ' ': 0xB9,
    'LS': 0x9D,
    'T': 0x94,
    'Z': 0xAC,
    '1': 0x82,
    '2': 0x83,
    '3': 0x84,
    '4': 0x85,
}

# SCANCODE of KOI(key of interest)
SC_idx2sc = [ key2scancode[k] for k in SN_idx2key[:-4] ]

SC_sc2idx = dict(zip(SC_idx2sc, range(STATE_DIM)))

CAMERA_NAMES = [
    'rgb',
    # 'alpha'
]


import os

# Task config
ROOT_PATH = os.path.dirname(__file__)
TASK_CONFIG = {
    'test': {
        # 'dataset_dir': os.path.join(__file__, './build/test'),
        # 需要__file__的parent
        'dataset_dir': os.path.join(ROOT_PATH, './build/test'),
        'num_episodes': 2,
        'episode_len': 400, # TODO: fix in record
        'camera_names': CAMERA_NAMES,
    },
    # 名椎滩好感任务
    'nazuchi_beach_friendship': {
        'dataset_dir': os.path.join(ROOT_PATH, './datasets/nazuchi_beach_friendship'),
        'num_episodes': 50, 
        'episode_len': 400, # TODO: fix in record
        # episode 现在是load_data运行时决定的
        # 推理时如何确定？
        'camera_names': CAMERA_NAMES,
    },
    # 名椎滩好感任务 + 不过于剧烈的随机视角，尚未录制
    # mr means mild random
    'nz2': {
        'dataset_dir': os.path.join(ROOT_PATH, 'F:/dataset/nz2'),
        'num_episodes': 28, 
        'episode_len': 400, # TODO: fix in record
        'camera_names': CAMERA_NAMES,
    }
}