from torch_geometric.data import HeteroData
from torch import nn
import torch
from ip.utils.common_utils import printarr, PositionalEncoder
from ip.utils.common_utils import SinusoidalPosEmb


class GraphRep(nn.Module):
    def __init__(self, config):
        super(GraphRep, self).__init__()
        ################################################################################################################
        # Store the parameters.
        self.batch_size = config['batch_size']
        self.num_demos = config['num_demos']
        self.traj_horizon = config['traj_horizon']
        self.num_scenes_nodes = config['num_scenes_nodes']
        self.num_freqs = config['local_num_freq']
        self.device = config['device']
        self.embd_dim = config['local_nn_dim']
        self.pred_horizon = config['pre_horizon']
        self.pos_in_nodes = config['pos_in_nodes']
        ################################################################################################################
        # These will be used to represent the gripper node positions.
        self.gripper_node_pos = torch.tensor([
            [0., 0., 0.],  # Middle of the gripper.
            [0., 0., -0.03],  # Tail of the gripper.
            [0., 0.03, 0.],  # Side of the gripper.
            [0., -0.03, 0.],  # Side of the gripper.
            [0., 0.03, 0.03],  # Finger.
            [0., -0.03, 0.03],  # finger.
        ], dtype=torch.float32, device=self.device) * 2
        self.num_g_nodes = len(self.gripper_node_pos)
        self.g_state_dim = 64
        self.d_time_dim = 64
        ################################################################################################################
        self.sine_pos_embd = SinusoidalPosEmb(self.d_time_dim)  # Think about this
        # Define the learnable embeddings for the nodes and edges.
        self.pos_embd = PositionalEncoder(3, self.num_freqs, log_space=True, add_original_x=True, scale=1.0)
        self.edge_dim = self.pos_embd.d_output * 2
        self.gripper_proj = nn.Linear(1, self.g_state_dim)

        self.gripper_embds = nn.Embedding(
            len(self.gripper_node_pos) * (self.pred_horizon + 1),
            self.embd_dim - self.g_state_dim, device=self.device)
        self.gripper_cond_gripper_embds = nn.Embedding(1, self.edge_dim, device=self.device)
        self.gripper_da_gripper_embds = nn.Embedding(1, self.edge_dim, device=self.device)
        ################################################################################################################
        # Define the structure of the graph.
        self.node_types = ['scene', 'gripper']
        self.edge_types = [
            # Local observation subgraphs.
            ('scene', 'rel', 'scene'),
            ('scene', 'rel', 'gripper'),
            ('gripper', 'rel', 'gripper'),
            # Propagation from demo gripper to current gripper.
            ('gripper', 'cond', 'gripper'),
            # Propagating information about the timestep in the demo.
            ('gripper', 'demo', 'gripper'),
            # Propagating information about the timestep in the demo.
            ('gripper', 'time_action', 'gripper'),
            # Propagating information from demo to action.
            ('gripper', 'demo_action', 'gripper'),
        ]
        self.graph = None
        ################################################################################################################

    def create_dense_edge_idx(self, num_nodes_source, num_nodes_dest):
        return torch.cartesian_prod(
            torch.arange(num_nodes_source, dtype=torch.int64, device=self.device),
            torch.arange(num_nodes_dest, dtype=torch.int64, device=self.device)).contiguous().t()

    def get_node_info(self):
        # A bunch of arange operations to store information which node in the graph belongs to which batch, timestep etc.
        # First the scene nodes. [bs, nd, th, sn, 3] + [bs, sn, 3]
        ################################################################################################################
        sb = torch.arange(self.batch_size, device=self.device)
        scene_batch = sb[:, None, None, None].repeat(1,
                                                     self.num_demos,
                                                     self.traj_horizon,
                                                     self.num_scenes_nodes
                                                     ).view(-1)
        sb_current = sb[:, None].repeat(1, self.num_scenes_nodes).view(-1)
        scene_batch = torch.cat([scene_batch, sb_current], dim=0)

        scene_traj = torch.arange(self.traj_horizon,
                                  device=self.device)[None, None, :, None].repeat(self.batch_size,
                                                                                  self.num_demos,
                                                                                  1,
                                                                                  self.num_scenes_nodes
                                                                                  ).view(-1)
        scene_traj = torch.cat([scene_traj, self.traj_horizon * torch.ones_like(sb_current)], dim=0)

        scene_demo = torch.arange(self.num_demos, device=self.device)[None, :, None, None].repeat(
            self.batch_size, 1, self.traj_horizon, self.num_scenes_nodes).view(-1)
        scene_current = self.num_demos * torch.ones(self.batch_size * self.num_scenes_nodes, device=self.device)
        scene_demo = torch.cat([scene_demo, scene_current], dim=0)

        # Accounting for scene action nodes.
        scene_batch_action = sb[:, None, None].repeat(1, self.pred_horizon, self.num_scenes_nodes).view(-1)
        scene_batch = torch.cat([scene_batch, scene_batch_action], dim=0)

        scene_traj_action = torch.arange(self.pred_horizon, device=self.device)[None, :, None].repeat(
            self.batch_size, 1, self.num_scenes_nodes).view(-1) + self.traj_horizon + 1
        scene_traj = torch.cat([scene_traj, scene_traj_action], dim=0)
        scene_demo = torch.cat([scene_demo, self.num_demos * torch.ones_like(scene_traj_action)], dim=0)
        ################################################################################################################
        # Now the gripper nodes. [bs, nd, th, gn, 3] + [bs, gn, 3] + [bs, ph, gn, 3]
        gripper_batch = sb[:, None, None, None].repeat(1, self.num_demos, self.traj_horizon, self.num_g_nodes).view(-1)
        gripper_batch_current = sb[:, None].repeat(1, self.num_g_nodes).view(-1)
        gripper_batch_action = sb[:, None, None].repeat(1, self.pred_horizon, self.num_g_nodes).view(-1)
        gripper_batch = torch.cat([gripper_batch, gripper_batch_current, gripper_batch_action], dim=0)

        gripper_time = torch.arange(self.traj_horizon, device=self.device, dtype=torch.long)[None, None, :,
                       None].repeat(self.batch_size, self.num_demos, 1, self.num_g_nodes).view(-1)

        gripper_time_current = self.traj_horizon * torch.ones(self.batch_size * self.num_g_nodes, device=self.device,
                                                              dtype=torch.long)
        gripper_time_action = torch.arange(self.pred_horizon, device=self.device, dtype=torch.long)[None, :,
                              None].repeat(self.batch_size, 1, self.num_g_nodes).view(-1)
        gripper_time = torch.cat([gripper_time,
                                  gripper_time_current,
                                  gripper_time_action + self.traj_horizon + 1], dim=0)

        gripper_node = torch.arange(self.num_g_nodes, device=self.device)[None, None, None, :].repeat(
            self.batch_size, self.num_demos, self.traj_horizon, 1).view(-1)
        gripper_node_current = torch.arange(self.num_g_nodes, device=self.device)[None, :].repeat(
            self.batch_size, 1).view(-1)
        gripper_node_action = torch.arange(self.num_g_nodes, device=self.device)[None, None, :].repeat(
            self.batch_size, self.pred_horizon, 1).view(-1)
        gripper_node = torch.cat([gripper_node, gripper_node_current, gripper_node_action], dim=0)

        gripper_emdb = gripper_node
        gripper_emdb[gripper_time > self.traj_horizon] += self.num_g_nodes * gripper_time_action

        gripper_demo = torch.arange(self.num_demos, device=self.device)[None, :, None, None].repeat(
            self.batch_size, 1, self.traj_horizon, self.num_g_nodes).view(-1)
        gripper_current = self.num_demos * torch.ones(self.batch_size * (self.pred_horizon + 1) * self.num_g_nodes,
                                                      device=self.device)
        gripper_demo = torch.cat([gripper_demo, gripper_current], dim=0)

        return {
            'scene': {
                'batch': scene_batch,
                'traj': scene_traj,
                'demo': scene_demo,
            },
            'gripper': {
                'batch': gripper_batch,
                'time': gripper_time,
                'node': gripper_node,
                'embd': gripper_emdb,
                'demo': gripper_demo,
            }
        }

    def transform_gripper_nodes(self, gripper_nodes, T):
        # gripper_nodes - [B, D, T, N, 3]
        # T - [B, D, T, 4, 4]
        has_demo = len(gripper_nodes.shape) == 5
        if not has_demo:
            gripper_nodes = gripper_nodes.unsqueeze(1)
        b, d, t, n, _ = gripper_nodes.shape
        gripper_nodes = gripper_nodes.reshape(-1, gripper_nodes.shape[-2], gripper_nodes.shape[-1]).permute(0, 2, 1)
        gripper_nodes = torch.bmm(T[..., :3, :3].reshape(-1, 3, 3), gripper_nodes)
        gripper_nodes += T[..., :3, 3].reshape(-1, 3, 1)
        gripper_nodes = gripper_nodes.permute(0, 2, 1).view(b, d, t, n, 3)
        if not has_demo:
            gripper_nodes = gripper_nodes.squeeze(1)
        return gripper_nodes

    def initialise_graph(self):
        # Manually connecting different nodes in the graph to achieve our desired graph representation.
        # Probably could be re-written to be more beautiful. Most definitely could.
        self.graph = HeteroData()
        node_info = self.get_node_info()

        dense_g_g = self.create_dense_edge_idx(node_info['gripper']['embd'].shape[0],
                                               node_info['gripper']['embd'].shape[0])

        dense_s_s = self.create_dense_edge_idx(node_info['scene']['batch'].shape[0],
                                               node_info['scene']['batch'].shape[0])

        dense_s_g = self.create_dense_edge_idx(node_info['scene']['batch'].shape[0],
                                               node_info['gripper']['embd'].shape[0])
        ################################################################################################################
        s_rel_s_mask = node_info['scene']['batch'][dense_s_s[0, :]] == node_info['scene']['batch'][dense_s_s[1, :]]
        s_rel_s_mask = s_rel_s_mask & (
                node_info['scene']['traj'][dense_s_s[0, :]] == node_info['scene']['traj'][dense_s_s[1, :]])
        s_rel_s_mask = s_rel_s_mask & (
                node_info['scene']['demo'][dense_s_s[0, :]] == node_info['scene']['demo'][dense_s_s[1, :]])
        ################################################################################################################
        s_rel_s_action_mask = s_rel_s_mask & (
                node_info['scene']['traj'][dense_s_s[0, :]] > self.traj_horizon)
        s_rel_s_action_mask = s_rel_s_action_mask & (
                node_info['scene']['traj'][dense_s_s[1, :]] > self.traj_horizon)
        s_rel_s_mask_demo = s_rel_s_mask & torch.logical_not(s_rel_s_action_mask)
        ################################################################################################################
        g_rel_g_mask = node_info['gripper']['batch'][dense_g_g[0, :]] == node_info['gripper']['batch'][dense_g_g[1, :]]
        g_rel_g_mask = g_rel_g_mask & (
                node_info['gripper']['time'][dense_g_g[0, :]] == node_info['gripper']['time'][dense_g_g[1, :]])
        g_rel_g_mask = g_rel_g_mask & (
                node_info['gripper']['demo'][dense_g_g[0, :]] == node_info['gripper']['demo'][dense_g_g[1, :]])
        ################################################################################################################
        s_rel_g_mask = node_info['scene']['batch'][dense_s_g[0, :]] == node_info['gripper']['batch'][dense_s_g[1, :]]
        s_rel_g_mask = s_rel_g_mask & (
                node_info['scene']['traj'][dense_s_g[0, :]] == node_info['gripper']['time'][dense_s_g[1, :]])
        s_rel_g_mask = s_rel_g_mask & (
                node_info['scene']['demo'][dense_s_g[0, :]] == node_info['gripper']['demo'][dense_s_g[1, :]])
        ################################################################################################################
        s_rel_g_action_mask = s_rel_g_mask & (
                node_info['scene']['traj'][dense_s_g[0, :]] > self.traj_horizon)
        s_rel_g_action_mask = s_rel_g_action_mask & (
                node_info['gripper']['time'][dense_s_g[1, :]] > self.traj_horizon)
        s_rel_g_mask_demo = s_rel_g_mask & torch.logical_not(s_rel_g_action_mask)
        ################################################################################################################
        g_c_g_mask = node_info['gripper']['batch'][dense_g_g[0, :]] == node_info['gripper']['batch'][dense_g_g[1, :]]
        g_c_g_mask = g_c_g_mask & (
                node_info['gripper']['time'][dense_g_g[0, :]] < self.traj_horizon)
        g_c_g_mask = g_c_g_mask & (node_info['gripper']['time'][dense_g_g[1, :]] == self.traj_horizon)
        ################################################################################################################
        g_t_g_mask = node_info['gripper']['batch'][dense_g_g[0, :]] == node_info['gripper']['batch'][dense_g_g[1, :]]
        g_t_g_mask = g_t_g_mask & (node_info['gripper']['time'][dense_g_g[0, :]] >= self.traj_horizon)
        g_t_g_mask = g_t_g_mask & (node_info['gripper']['time'][dense_g_g[1, :]] > self.traj_horizon)
        g_t_g_mask = g_t_g_mask & (
                node_info['gripper']['time'][dense_g_g[1, :]] != node_info['gripper']['time'][dense_g_g[0, :]])
        g_tc_g = g_t_g_mask & (node_info['gripper']['time'][dense_g_g[0, :]] == self.traj_horizon)
        g_t_g_mask = g_t_g_mask & torch.logical_not(g_tc_g)
        ################################################################################################################
        g_d_g_mask = node_info['gripper']['batch'][dense_g_g[0, :]] == node_info['gripper']['batch'][dense_g_g[1, :]]
        g_d_g_mask = g_d_g_mask & (node_info['gripper']['time'][dense_g_g[0, :]] < self.traj_horizon)
        g_d_g_mask = g_d_g_mask & (node_info['gripper']['time'][dense_g_g[1, :]] < self.traj_horizon)
        g_d_g_mask = g_d_g_mask & (
                node_info['gripper']['time'][dense_g_g[0, :]] != node_info['gripper']['time'][dense_g_g[1, :]])
        g_d_g_mask = g_d_g_mask & (
                node_info['gripper']['demo'][dense_g_g[0, :]] == node_info['gripper']['demo'][dense_g_g[1, :]])
        g_d_g_mask = g_d_g_mask & (node_info['gripper']['time'][dense_g_g[1, :]] - node_info['gripper']['time'][
            dense_g_g[0, :]] == -1)
        ################################################################################################################
        self.graph.gripper_batch = node_info['gripper']['batch']
        self.graph.gripper_time = node_info['gripper']['time']
        self.graph.gripper_node = node_info['gripper']['node']
        self.graph.gripper_embd = node_info['gripper']['embd'].long()
        self.graph.gripper_demo = node_info['gripper']['demo']
        self.graph.scene_batch = node_info['scene']['batch']
        self.graph.scene_traj = node_info['scene']['traj']
        self.graph.scene_demo = node_info['scene']['demo']

        self.graph[('gripper', 'rel', 'gripper')].edge_index = dense_g_g[:, g_rel_g_mask]
        self.graph[('scene', 'rel', 'scene')].edge_index = dense_s_s[:, s_rel_s_mask]
        self.graph[('scene', 'rel', 'gripper')].edge_index = dense_s_g[:, s_rel_g_mask]
        self.graph[('gripper', 'cond', 'gripper')].edge_index = dense_g_g[:, g_c_g_mask]
        self.graph[('gripper', 'time_action', 'gripper')].edge_index = dense_g_g[:, g_t_g_mask]
        self.graph[('gripper', 'demo', 'gripper')].edge_index = dense_g_g[:, g_d_g_mask]

        self.graph[('scene', 'rel_action', 'gripper')].edge_index = dense_s_g[:, s_rel_g_action_mask]
        self.graph[('scene', 'rel_demo', 'gripper')].edge_index = dense_s_g[:, s_rel_g_mask_demo]
        self.graph[('scene', 'rel_action', 'scene')].edge_index = dense_s_s[:, s_rel_s_action_mask]
        self.graph[('scene', 'rel_demo', 'scene')].edge_index = dense_s_s[:, s_rel_s_mask_demo]
        self.graph[('gripper', 'rel_cond', 'gripper')].edge_index = dense_g_g[:, g_tc_g]

    def update_graph(self, data):
        # Adding information to the graph structure create in initialise_graph.
        # scene_node_pos: # [B, N, T, S, 3]
        gripper_node_pos = self.gripper_node_pos[None, None, None, :, :].repeat(self.batch_size,
                                                                                self.num_demos,
                                                                                self.traj_horizon, 1, 1)
        ################################################################################################################
        # demo_T_w_es: [B, D, T, 4, 4]
        # T_w_e: [B, 4, 4]
        # T_w_n: [B, P, 4, 4]
        # Create identity matrix like T_w_e
        I_w_e = torch.eye(4, device=self.device)[None, :, :].repeat(self.batch_size, 1, 1)

        all_T_w_e = torch.cat([
            data.demo_T_w_es[:, :self.num_demos, :, None, :, :].repeat(1, 1, 1, 6, 1, 1).view(-1, 4, 4),
            I_w_e[:, None, :, :].repeat(1, 6, 1, 1).view(-1, 4, 4),
            data.actions[:, :, None, :, :].repeat(1, 1, 6, 1, 1).view(-1, 4, 4)
        ])
        all_T_e_w = all_T_w_e.inverse()
        ################################################################################################################

        gripper_node_pos_current = gripper_node_pos[:, 0, 0, ...].view(self.batch_size, -1, 3)
        gripper_node_pos_action = self.gripper_node_pos[None, None, :, :].repeat(self.batch_size,
                                                                                 self.pred_horizon, 1, 1)

        gripper_node_pos = torch.cat([gripper_node_pos.reshape(-1, 3),
                                      gripper_node_pos_current.reshape(-1, 3),
                                      gripper_node_pos_action.reshape(-1, 3)], dim=0)

        # data.graps_demos [B, D, T, 1]
        gripper_states = self.gripper_proj(data.graps_demos[:, :self.num_demos])[..., None, :].repeat(1, 1, 1,
                                                                                                      self.num_g_nodes,
                                                                                                      1)
        gripper_states = gripper_states.view(-1, self.g_state_dim)
        gripper_states_current = self.gripper_proj(data.current_grip.unsqueeze(-1))[..., None, :].repeat(1,
                                                                                                         self.num_g_nodes,
                                                                                                         1)
        gripper_states_current = gripper_states_current.view(-1, self.g_state_dim)
        gripper_states_action = self.gripper_proj(data.actions_grip.unsqueeze(-1))[..., None, :].repeat(1, 1,
                                                                                                        self.num_g_nodes,
                                                                                                        1)
        gripper_states_action = gripper_states_action.view(-1, self.g_state_dim)
        gripper_states = torch.cat([gripper_states, gripper_states_current, gripper_states_action], dim=0)
        gripper_embd = self.gripper_embds(self.graph.gripper_embd)

        # Adding diffusion time step information to gripper action nodes.
        d_time_embd = self.sine_pos_embd(data.diff_time)[:, None, ...].repeat(1,
                                                                              self.pred_horizon,
                                                                              self.num_g_nodes,
                                                                              1).view(-1, self.d_time_dim)
        gripper_embd[self.graph.gripper_time > self.traj_horizon][:, -self.d_time_dim:] = d_time_embd

        gripper_embd = torch.cat([gripper_embd, gripper_states], dim=-1)

        scene_node_pos = torch.cat([
            data.demo_scene_node_pos[:, :self.num_demos].reshape(-1, 3),
            data.live_scene_node_pos.view(-1, 3),
            data.action_scene_node_pos.view(-1, 3)
        ], dim=0)
        scene_node_embd = torch.cat([
            data.demo_scene_node_embds[:, :self.num_demos].reshape(-1, self.embd_dim),
            data.live_scene_node_embds.view(-1, self.embd_dim),
            data.action_scene_node_embds.view(-1, self.embd_dim)
        ], dim=0)

        self.graph['gripper'].pos = gripper_node_pos
        self.graph['gripper'].x = gripper_embd
        self.graph['scene'].pos = scene_node_pos
        self.graph['scene'].x = scene_node_embd

        if self.pos_in_nodes:
            self.graph['gripper'].x = \
                torch.cat([self.graph['gripper'].x, self.pos_embd(self.graph['gripper'].pos)], dim=-1)
            self.graph['scene'].x = \
                torch.cat([self.graph['scene'].x, self.pos_embd(self.graph['scene'].pos)], dim=-1)

        self.add_rel_edge_attr('scene', 'gripper')
        self.add_rel_edge_attr('gripper', 'gripper')
        self.add_rel_edge_attr('scene', 'scene')

        self.graph[('gripper', 'cond', 'gripper')].edge_attr = self.gripper_cond_gripper_embds(
            torch.zeros(len(self.graph[('gripper', 'cond', 'gripper')].edge_index[0]), device=self.device).long())

        self.add_rel_edge_attr('scene', 'gripper', edge='rel_action')
        self.add_rel_edge_attr('scene', 'gripper', edge='rel_demo')

        self.add_rel_edge_attr('scene', 'scene', edge='rel_demo')
        self.add_rel_edge_attr('scene', 'scene', edge='rel_action')

        self.add_rel_edge_attr('gripper', 'gripper', edge='time_action',
                               all_T_w_e=all_T_w_e, all_T_e_w=all_T_e_w)
        self.add_rel_edge_attr('gripper', 'gripper', edge='rel_cond',
                               all_T_w_e=all_T_w_e, all_T_e_w=all_T_e_w)
        self.add_rel_edge_attr('gripper', 'gripper', edge='demo',
                               all_T_w_e=all_T_w_e, all_T_e_w=all_T_e_w)

    def add_rel_edge_attr(self, source, dest, edge='rel', all_T_w_e=None, all_T_e_w=None):
        if all_T_w_e is None:
            pos_dest = self.graph[dest].pos[self.graph[(source, edge, dest)].edge_index[1]]
            pos_source = self.graph[source].pos[self.graph[(source, edge, dest)].edge_index[0]]
            pos_dest_rot = pos_dest
        else:
            pos_source = self.graph[source].pos[self.graph[(source, edge, dest)].edge_index[0]]
            T_i_j = torch.bmm(all_T_e_w[self.graph[(source, edge, dest)].edge_index[0]],
                              all_T_w_e[self.graph[(source, edge, dest)].edge_index[1]])
            pos_dest_rot = torch.bmm(T_i_j[..., :3, :3], pos_source[..., None]).squeeze(-1)
            pos_dest = pos_source + T_i_j[..., :3, 3]
        self.graph[(source, edge, dest)].edge_attr = torch.cat([self.pos_embd(pos_dest - pos_source),
                                                                self.pos_embd(pos_dest_rot - pos_source)], dim=-1)
