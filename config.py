import os

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

    # srds，这个仅在推理时使用
    'temporal_agg': True,

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

    'temporal_agg': True,

    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
}


configs = {
    'mlp': __mlp_config,
    'act': __act_config,
}