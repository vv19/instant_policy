import torch
import numpy as np

config = {
    'record': False,
    'save_dir': None,
    'scene_encoder_path': './checkpoints/scene_encoder.pt',
    'pre_trained_encoder': True,
    'freeze_encoder': True,
    'save_every': 100000,
    'compile_models': False,
    # Model config
    'local_num_freq': 10,
    'local_nn_dim': 512,
    'hidden_dim': 1024,
    'num_demos': 2,
    'randomise_num_demos': False,
    'num_demos_test': 2,
    'traj_horizon': 10,
    'device': 'cuda',
    'batch_size': 16,
    'batch_size_val': 1,
    'num_scenes_nodes': 16,
    'pre_horizon': 8,
    'pos_in_nodes': True,
    'num_layers': 2,

    # Diffusion config
    'lr': 1e-5,
    'weight_decay': 1e-2,
    'use_lr_scheduler': False,
    'num_warmup_steps': 1000,
    'num_diffusion_iters_train': 100,
    'num_diffusion_iters_test': 8,
    'num_iters': 50000000001,

    'test_every': 50000,
    'randomize_g_prob': 0.1,

    'min_actions': torch.tensor([-0.01] * 3 + [-np.deg2rad(3), -np.deg2rad(3), -np.deg2rad(3)], dtype=torch.float32),
    'max_actions': torch.tensor([0.01] * 3 + [np.deg2rad(3), np.deg2rad(3), np.deg2rad(3)], dtype=torch.float32),
}
