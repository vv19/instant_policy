from torch.utils.data import Dataset
import torch
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class RunningDataset(Dataset):
    def __init__(self, data_path, num_samples, rec=False, rand_g_prob=0.0, random_rotation=False):
        self.data_path = data_path
        self.num_samples = num_samples
        self.rand_g_prob = rand_g_prob
        self.random_rotation = random_rotation
        self.rec = rec
        if rec:
            self.data_attr = [
                'pos',
                'queries',
                'batch_queries',
                'batch_pos',
                'occupancy',
            ]
        else:
            self.data_attr = [
                # 'pos_demos',
                # 'graps_demos',
                # 'batch_demos',
                # 'pos_obs',
                # 'current_grip',
                # 'batch_pos_obs',
                # 'past_actions',
                # 'past_actions_grip',
                'actions',
                'actions_grip',
            ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        while True:
            try:
                data = torch.load(os.path.join(self.data_path, 'data_{}.pt'.format(idx)))
                # Make sure that the data has all the required attributes.
                for attr in self.data_attr:
                    assert hasattr(data, attr)

                if np.random.uniform() < self.rand_g_prob:
                    data.current_grip *= -1

                if self.random_rotation and self.rec:
                    R = torch.tensor(Rot.random().as_matrix(), dtype=data.pos.dtype, device=data.pos.device)
                    data.pos = data.pos @ R.T
                    data.queries = data.queries @ R.T
                return data
            except Exception as e:
                idx = np.random.randint(0, self.num_samples)
