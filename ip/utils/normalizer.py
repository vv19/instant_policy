import torch


class Normalizer:
    def __init__(self, pred_horizon, min_action, max_action, gripper_length=0.06, device='cuda'):
        self.pred_horizon = pred_horizon

        self.min_action = min_action[None, None, :].repeat(1, pred_horizon, 1)
        self.max_action = max_action[None, None, :].repeat(1, pred_horizon, 1)

        # Assumes that min_action are negative and max_action are positive
        # Scale 1 dim by the corresponding time step
        self.min_action *= torch.linspace(1, pred_horizon, pred_horizon, device=device, dtype=torch.float)[None, :,
                           None]
        self.max_action *= torch.linspace(1, pred_horizon, pred_horizon, device=device, dtype=torch.float)[None, :,
                           None]

        max_angle = self.max_action[..., -1]
        delta_rot = torch.sqrt(2 * gripper_length ** 2 - 2 * gripper_length ** 2 * torch.cos(max_angle))
        delta_rot = delta_rot[..., None, None].repeat(1, 1, 1, 3)

        self.min_labels = (2 * self.min_action[..., :3]).unsqueeze(2)
        self.max_labels = (2 * self.max_action[..., :3]).unsqueeze(2)

        self.min_labels = torch.cat([self.min_labels[..., :3], -2 * delta_rot], dim=-1)
        self.max_labels = torch.cat([self.max_labels[..., :3], 2 * delta_rot], dim=-1)

    def normalize_actions(self, actions):
        '''
        Normalize actions to [-1, 1]
        :param actions : (bs, pred_horizon, 6)
        :return: normalized actions : (bs, pred_horizon, 6)
        '''
        return 2 * (actions - self.min_action) / (self.max_action - self.min_action) - 1

    def denormalize_actions(self, actions):
        '''
        Denormalize actions to the original range
        :param actions : (bs, pred_horizon, 6)
        :return: denormalized actions : (bs, pred_horizon, 6)
        '''
        return 0.5 * (actions + 1) * (self.max_action - self.min_action) + self.min_action

    def normalize_labels(self, labels):
        '''
        Normalize labels to [-1, 1]
        :param labels : (bs, pred_horizon, 3)
        :return: normalized labels : (bs, pred_horizon, 3)
        '''
        return 2 * (labels - self.min_labels) / (self.max_labels - self.min_labels) - 1

    def denormalize_labels(self, labels):
        '''
        Denormalize labels to the original range
        :param labels : (bs, pred_horizon, 3)
        :return: denormalized labels : (bs, pred_horizon, 3)
        '''
        return 0.5 * (labels + 1) * (self.max_labels - self.min_labels) + self.min_labels
