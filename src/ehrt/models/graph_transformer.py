import torch
from torch import nn
import torch_geometric.nn as tgmnn

from src.ehrt.transformer_conv import TransformerConv


class GraphTransformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv = tgmnn.Sequential('x, edge_index, edge_attr, batch', [
            (TransformerConv(config.hidden_size // 5, config.hidden_size // 5, heads=2, edge_dim=config.hidden_size // 5, dropout=config.hidden_dropout_prob, concat=True), 'x, edge_index, edge_attr -> x'),
            nn.GELU(),
            (TransformerConv(config.hidden_size // 5, config.hidden_size // 5, heads=2, edge_dim=config.hidden_size // 5, dropout=config.hidden_dropout_prob, concat=True), 'x, edge_index, edge_attr -> x'),
            nn.GELU(),
            (TransformerConv(config.hidden_size // 5, config.hidden_size // 5, heads=2, edge_dim=config.hidden_size // 5, dropout=config.hidden_dropout_prob, concat=False), 'x, edge_index, edge_attr -> x'),
        ])

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size // 5)
        self.embed_ee = nn.Embedding(7, config.hidden_size // 5)

    def forward(self, x, edge_index, edge_index_readout, edge_attr, batch):
        indices = (x==0).nonzero().squeeze()
        h_nodes = self.conv(self.embed(x), edge_index, self.embed_ee(edge_attr), batch)
        x = h_nodes[indices]
        return x