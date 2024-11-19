import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import to_hetero
from torch_geometric.nn import MLP


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1, edge_dim=3, num_layers=2, metadata=None, dropout=0.3,
                 norm=None):
        super().__init__()
        in_channels_list = [in_channels] + [hidden_channels] * (num_layers - 1)
        self.transformer_blocks = nn.ModuleList(
            [to_hetero(
                GraphTransformerBlock(in_channels_list[i], hidden_channels, heads, edge_dim=edge_dim, dropout=dropout,
                                      norm=norm),
                metadata=metadata,
                aggr='sum') for i in range(num_layers)
            ])

    def forward(self, x, edge_index, edge_attr):
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, edge_index, edge_attr)
        return x


class GraphTransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_heads=1, edge_dim=None, dropout=0.3, norm=None):
        super().__init__()

        self.att = TransformerConv(in_channels, hidden_channels // n_heads, edge_dim=edge_dim, heads=n_heads,
                                   concat=True, dropout=dropout, bias=True, root_weight=True)
        self.lin = MLP([hidden_channels, hidden_channels, hidden_channels],
                       act=nn.GELU(approximate='tanh'),
                       plain_last=False,
                       norm=norm
                       )

    def forward(self, x, edge_index, edge_attr):
        x = self.att(x, edge_index, edge_attr)
        x = x + self.lin(x)
        return x
