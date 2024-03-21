import os
from copy import deepcopy

__mlp_config = {
    'device': 'cuda',

    'policy_class': 'mlp',
    'ckpt_dir': os.path.join(os.path.dirname(__file__), 'models/mlp'),
    'pretrained': True,
    'pretrained_ckpt': r'./models/mlp/policy_epoch_100_seed_0.ckpt',
    'task_name': 'nazuchi_beach_friendship',
    'batch_size': 20,
    'seed': 0,
    'num_epochs': 2000,
    'lr': 1e-4,
    # mlp 目前参数量在config里面没法调，都是写死的
    # 实际没用
    'kl_weight': 100,
    'chunk_size': 20,
    'hidden_dim': 64,
    'dim_feedforward': 1280,

    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
}

__act_config = {
    'device': 'cuda',

    'policy_class': 'act',  # 'mlp' or 'act for now
    'ckpt_dir': os.path.join(os.path.dirname(__file__), 'models/act_100l1'),
    'pretrained': True,
    'pretrained_ckpt': r'./models/act_100l1/policy_epoch_600_seed_0.ckpt',
    'task_name': 'nazuchi_beach_friendship',
    'batch_size': 25,
    'seed': 0,
    'num_epochs': 1400,
    'lr': 1e-4,
    'kl_weight': 100,
    'chunk_size': 40,
    'hidden_dim': 640,
    'dim_feedforward': 1280,

    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
}

__mlp_infer_config = {**__mlp_config, **{
    'seed': 1000,
    'ckpt_name': 'policy_best.ckpt',
    'temporal_agg': True,
    # 'onscreen_render': True,
    # 'save_video': True,
    'video_dir': os.path.join(os.path.dirname(__file__), 'video/mlp'),
}}

__act_infer_config = {**__act_config, **{
    'ckpt_dir': os.path.join(os.path.dirname(__file__), 'models/act_after6k'),
    'seed': 1000,
    # 'ckpt_name': 'policy_best.ckpt',
    'ckpt_name': 'policy_epoch_2300_seed_0.ckpt',
    'temporal_agg': True,
    # 'onscreen_render': True,
    # 'save_video': True,
    'video_dir': os.path.join(os.path.dirname(__file__), 'video/act'),
}}


configs = {
    'mlp': __mlp_config,
    'act': __act_config,
}

infer_configs = {
    'mlp': __mlp_infer_config,
    'act': __act_infer_config,
}