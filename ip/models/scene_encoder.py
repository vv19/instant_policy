import torch
from torch_geometric.nn import fps, nearest
from torch_geometric.nn import PointNetConv
from ip.utils.common_utils import PositionalEncoder
from torch_geometric.nn import MLP
from typing import Optional, Union
from torch_geometric.typing import OptTensor, PairOptTensor, Adj, PairTensor, Tensor


class SceneEncoder(torch.nn.Module):
    def __init__(self, num_freqs, embd_dim=256):
        super().__init__()
        self.embd_dim = embd_dim
        # Parameters are just vibes.
        self.sa1_module = SAModule(0.125, [3, 128, 128, 128],
                                   global_nn_dims=[128, 256, 256],
                                   num_freqs=num_freqs,
                                   scale=1 / 0.05,
                                   norm=None)
        self.sa2_module = SAModule(0.0625, [256 + 3, 512, 512, 512],
                                   global_nn_dims=[512, 512, embd_dim],
                                   num_freqs=num_freqs,
                                   scale=1 / 0.2,
                                   norm=None,
                                   plain_last=True)

    def forward(self, x, pos, batch):
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        return sa2_out


class PointNetConvPE(PointNetConv):
    def __init__(self, nn_dims, global_nn_dims=None, aggr='mean', num_freqs=4, cat_pos=False,
                 scale=1., plain_last=False, norm=None):
        self.scale = scale

        # Adjust nn_dims to include positional encoding.
        nn_dims[0] += 3 * (2 * num_freqs)
        nn = MLP(nn_dims, norm=None, act=torch.nn.GELU(approximate='tanh'), plain_last=False)

        if cat_pos and global_nn_dims is not None:
            global_nn_dims[0] += 3 * (2 * num_freqs + 1)

        global_nn = None if global_nn_dims is None else \
            MLP(global_nn_dims, norm=None, act=torch.nn.GELU(approximate='tanh'),
                plain_last=plain_last)

        self.cat_pos = cat_pos
        super().__init__(nn, global_nn=global_nn, add_self_loops=False, aggr=aggr)
        self.pe = PositionalEncoder(3, num_freqs)

    def message(self, x_j: Optional[Tensor], pos_i: Tensor,
                pos_j: Tensor) -> Tensor:
        msg = self.pe((pos_j - pos_i) * self.scale)
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:

        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        # propagate_type: (x: PairOptTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, size=None)

        if self.global_nn is not None:
            if self.cat_pos:
                out = torch.cat([out, self.pe(pos[1])], dim=1)
            out = self.global_nn(out)

        return out


class SAModule(torch.nn.Module):
    def __init__(self, ratio, nn_dims, global_nn_dims=None, num_freqs=4, aggr='mean', cat_pos=False,
                 scale=1., plain_last=False, norm=None):
        super().__init__()
        self.cat_pos = cat_pos
        self.ratio = ratio
        self.conv = PointNetConvPE(nn_dims, global_nn_dims, aggr=aggr, num_freqs=num_freqs, cat_pos=cat_pos,
                                   scale=scale, plain_last=plain_last, norm=norm)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row = nearest(pos, pos[idx], batch, batch[idx])
        col = torch.arange(0, pos.shape[0], dtype=torch.long, device=pos.device)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return [x, pos, batch]
