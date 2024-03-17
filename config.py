import os
from copy import deepcopy

__mlp_config = {
    'device': 'cuda',

    'policy_class': 'mlp',
    'ckpt_dir': os.path.join(os.path.dirname(__file__), 'models/mlp'),
    'pretrained': False,
    'pretrained_ckpt': None,
    'task_name': 'nazuchi_beach_friendship',
    'batch_size': 4,
    'seed': 0,
    'num_epochs': 2000,
    'lr': 1e-4,
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
    'ckpt_dir': os.path.join(os.path.dirname(__file__), 'models/act'),
    'pretrained': False,
    'pretrained_ckpt': None,
    'task_name': 'nazuchi_beach_friendship',
    'batch_size': 4,
    'seed': 0,
    'num_epochs': 2000,
    'lr': 1e-4,
    'kl_weight': 100,
    'chunk_size': 20,
    'hidden_dim': 64,
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
    'seed': 1000,
    'ckpt_name': 'policy_best.ckpt',
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