import torch
import torch_geometric
import time
from ip.models.scene_encoder import SceneEncoder
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from ip.models.graph_transformer import GraphTransformer
from torch_geometric.nn import MLP
import torch.nn as nn
from ip.utils.common_utils import dfs_freeze
from ip.models.graph_rep import GraphRep


class AGI(torch.nn.Module):
    def __init__(self, config):
        super(AGI, self).__init__()
        self.config = config
        self.num_demos = config['num_demos']
        self.num_demos_in_use = config['num_demos']
        self.traj_horizon = config['traj_horizon']
        self.local_embd_dim = config['local_nn_dim']
        self.batch_size = config['batch_size']
        self.num_scenes_nodes = config['num_scenes_nodes']
        self.pred_horizon = config['pre_horizon']
        self.num_layers = config['num_layers']
        compile_models = config['compile_models']

        self.scene_encoder = SceneEncoder(num_freqs=10,
                                          embd_dim=config['local_nn_dim']).to(config['device'])
        if self.config['pre_trained_encoder']:
            self.scene_encoder.load_state_dict(torch.load(config['scene_encoder_path']))
            if self.config['freeze_encoder']:
                dfs_freeze(self.scene_encoder)

        self.graph = GraphRep(config)
        self.graph.initialise_graph()

        in_channels = self.local_embd_dim
        if config['pos_in_nodes']:
            in_channels += self.graph.edge_dim // 2

        self.local_encoder = GraphTransformer(in_channels=in_channels,
                                              hidden_channels=config['hidden_dim'],
                                              heads=config['hidden_dim'] // 64,
                                              num_layers=self.num_layers,
                                              metadata=(['scene', 'gripper'],
                                                        [
                                                            ('scene', 'rel', 'scene'),
                                                            ('scene', 'rel', 'gripper'),
                                                            ('gripper', 'rel', 'gripper'),
                                                        ]),
                                              edge_dim=self.graph.edge_dim,
                                              dropout=0.0,
                                              norm='layer').to(config['device'])

        self.cond_encoder = GraphTransformer(in_channels=config['hidden_dim'],
                                             hidden_channels=config['hidden_dim'],
                                             heads=config['hidden_dim'] // 64,
                                             num_layers=self.num_layers,
                                             metadata=(['gripper', 'scene'],
                                                       [
                                                           ('gripper', 'cond', 'gripper'),
                                                           ('gripper', 'demo', 'gripper'),
                                                           ('scene', 'rel_demo', 'gripper'),
                                                           ('scene', 'rel_demo', 'scene'),
                                                       ]),
                                             edge_dim=self.graph.edge_dim,
                                             dropout=0.0,
                                             norm='layer').to(config['device'])

        self.action_encoder = GraphTransformer(in_channels=config['hidden_dim'],
                                               hidden_channels=config['hidden_dim'],
                                               heads=config['hidden_dim'] // 64,
                                               num_layers=self.num_layers,
                                               metadata=(['gripper', 'scene'],
                                                         [
                                                             ('gripper', 'time_action', 'gripper'),
                                                             ('gripper', 'rel_cond', 'gripper'),
                                                             ('scene', 'rel_action', 'gripper'),
                                                             ('scene', 'rel_action', 'scene'),
                                                         ]),
                                               edge_dim=self.graph.edge_dim,
                                               dropout=0.0,
                                               norm='layer').to(config['device'])

        # Separate head for trans, rot and grip.
        self.prediction_head = MLP([config['hidden_dim'], self.local_embd_dim, 3], act='GELU',
                                   plain_last=True, norm='layer_norm')
        self.prediction_head_rot = MLP([config['hidden_dim'], self.local_embd_dim, 3], act='GELU',
                                       plain_last=True, norm='layer_norm')
        self.prediction_head_g = MLP([config['hidden_dim'], self.local_embd_dim, 1], act='GELU',
                                     plain_last=True, norm='layer_norm')

        if compile_models:
            self.compile_models()

    def reinit_graphs(self, batch_size, num_demos=None):
        self.batch_size = batch_size
        if num_demos is not None:
            self.num_demos = num_demos
            self.graph.num_demos = num_demos
        self.graph.batch_size = batch_size
        self.graph.initialise_graph()

    def compile_models(self):
        self.scene_encoder.sa1_module.conv = torch.compile(self.scene_encoder.sa1_module.conv, mode="reduce-overhead")
        self.scene_encoder.sa2_module.conv = torch.compile(self.scene_encoder.sa2_module.conv, mode="reduce-overhead")
        self.local_encoder = torch.compile(self.local_encoder, mode="reduce-overhead")
        self.cond_encoder = torch.compile(self.cond_encoder, mode="reduce-overhead")
        self.action_encoder = torch.compile(self.action_encoder, mode="reduce-overhead")
        self.prediction_head = torch.compile(self.prediction_head, mode="reduce-overhead")
        self.prediction_head_rot = torch.compile(self.prediction_head_rot, mode="reduce-overhead")
        self.prediction_head_g = torch.compile(self.prediction_head_g, mode="reduce-overhead")

    def get_labels(self, gt_actions, noisy_actions, gt_grips, noisy_grips, delta_grip=False, sep_rot=True):
        # gt_actions: (bs, pred_horizon, 4, 4)
        # noisy_actions: (bs, pred_horizon, 4, 4)
        # gt_grips: (bs, pred_horizon, 1)
        # noisy_grips: (bs, pred_horizon, 1)
        gripper_points = self.graph.gripper_node_pos[None, None, :].repeat(gt_actions.shape[0],
                                                                           gt_actions.shape[1], 1, 1)

        if sep_rot:
            T_w_n = noisy_actions.view(-1, 4, 4)
            T_n_w = torch.inverse(T_w_n)
            T_w_g = gt_actions.view(-1, 4, 4)
            T_n_g = torch.bmm(T_n_w, T_w_g)
            T_n_g = T_n_g.view(gt_actions.shape[0], gt_actions.shape[1], 4, 4)

            labels_trans = T_n_g[..., :3, 3][:, :, None, :].repeat(1, 1,
                                                                   gripper_points.shape[-2],
                                                                   1)
            T_n_g[..., :3, 3] = 0
            labels_rot = self.graph.transform_gripper_nodes(gripper_points, T_n_g) - gripper_points
            labels = torch.cat([labels_trans, labels_rot], dim=-1)
        else:
            gripper_points_gt = self.graph.transform_gripper_nodes(gripper_points, gt_actions)
            gripper_points_noisy = self.graph.transform_gripper_nodes(gripper_points, noisy_actions)
            labels = gripper_points_gt - gripper_points_noisy

        if delta_grip:
            labels_grip = gt_grips - noisy_grips
        else:
            labels_grip = gt_grips
        labels_grip = labels_grip[:, :, None, :].repeat(1, 1, gripper_points.shape[-2], 1)
        labels = torch.cat([labels, labels_grip], dim=-1)
        return labels

    def get_transformed_node_pos(self, actions, transform=True):
        gripper_points = self.graph.gripper_node_pos[None, None, :].repeat(actions.shape[0], actions.shape[1], 1, 1)
        if transform:
            gripper_points = self.graph.transform_gripper_nodes(gripper_points, actions)
        return gripper_points

    def forward(self, data):
        if not hasattr(data, 'demo_scene_node_embds'):
            data.demo_scene_node_embds, data.demo_scene_node_pos = self.get_demo_scene_emb(data)

        if not hasattr(data, 'live_scene_node_embds'):
            data.live_scene_node_embds, data.live_scene_node_pos = self.get_live_scene_emb(data)
        ################################################################################################################
        current_obs = to_dense_batch(data.pos_obs, data.batch_pos_obs, fill_value=0)[0]
        current_obs = current_obs[:, None, ...].repeat(1, self.pred_horizon, 1, 1)
        current_obs = current_obs.view(self.batch_size * self.pred_horizon, -1, 3)
        actions = data.actions.view(-1, 4, 4)

        current_obs = torch.bmm(actions[:, :3, :3].transpose(1, 2), current_obs.permute(0, 2, 1)).permute(0, 2, 1)
        current_obs -= actions[:, :3, 3][:, None, :]

        action_batch = torch.arange(current_obs.shape[0], device=current_obs.device)[:, None].repeat(1,
                                                                                                     current_obs.shape[
                                                                                                         1])
        action_batch = action_batch.view(-1)
        current_obs = current_obs.reshape(-1, 3)

        pos_obs_old = data.pos_obs.clone()
        batch_pos_obs_old = data.batch_pos_obs.clone()

        data.pos_obs = current_obs
        data.batch_pos_obs = action_batch

        action_scene_node_embds, action_scene_node_pos = self.get_live_scene_emb(data)

        data.pos_obs = pos_obs_old
        data.batch_pos_obs = batch_pos_obs_old

        data.action_scene_node_embds = action_scene_node_embds.view(self.batch_size, self.pred_horizon,
                                                                    -1, self.local_embd_dim)
        data.action_scene_node_pos = action_scene_node_pos.view(self.batch_size, self.pred_horizon, -1, 3)
        ################################################################################################################
        self.graph.update_graph(data)

        # TODO: This can cause problems for some GPU types when compiling, but it is needed for other types of GPUs.
        torch.compiler.cudagraph_mark_step_begin()

        x_dict = self.local_encoder(self.graph.graph.x_dict,
                                    self.graph.graph.edge_index_dict,
                                    self.graph.graph.edge_attr_dict)

        x_dict = self.cond_encoder(x_dict,
                                   self.graph.graph.edge_index_dict,
                                   self.graph.graph.edge_attr_dict)

        x_dict = self.action_encoder(x_dict,
                                     self.graph.graph.edge_index_dict,
                                     self.graph.graph.edge_attr_dict)
        ################################################################################################################
        x_gripper = x_dict['gripper'][self.graph.graph.gripper_time > self.traj_horizon].view(self.batch_size,
                                                                                              self.pred_horizon,
                                                                                              self.graph.num_g_nodes,
                                                                                              -1)

        preds_t = self.prediction_head(x_gripper)
        preds_rot = self.prediction_head_rot(x_gripper)
        preds_g = self.prediction_head_g(x_gripper)
        preds = torch.cat([preds_t, preds_rot, preds_g], dim=-1)
        return preds

    def get_demo_scene_emb(self, data):
        bs = data.actions.shape[0]
        demo_scene_node_embds, demo_scene_node_pos, demo_scene_node_batch = \
            self.scene_encoder(None,
                               data.pos_demos,
                               data.batch_demos)
        demo_scene_node_embds = to_dense_batch(demo_scene_node_embds, demo_scene_node_batch, fill_value=0)[0]
        demo_scene_node_embds = demo_scene_node_embds.view(bs, self.num_demos, self.traj_horizon, -1,
                                                           self.local_embd_dim)
        demo_scene_node_pos = to_dense_batch(demo_scene_node_pos, demo_scene_node_batch, fill_value=0)[0]
        demo_scene_node_pos = demo_scene_node_pos.view(bs, self.num_demos, self.traj_horizon, -1, 3)
        return demo_scene_node_embds, demo_scene_node_pos

    def get_live_scene_emb(self, data):
        current_scene_node_embds, current_scene_node_pos, current_scene_node_batch = \
            self.scene_encoder(None,
                               data.pos_obs,
                               data.batch_pos_obs)
        current_scene_node_embds = to_dense_batch(current_scene_node_embds, current_scene_node_batch, fill_value=0)[0]
        current_scene_node_pos = to_dense_batch(current_scene_node_pos, current_scene_node_batch, fill_value=0)[0]
        return current_scene_node_embds, current_scene_node_pos
