import torch
import torch.nn as nn
import torch_geometric.nn as gn

try:
    PointConv = gn.PointConv
except AttributeError:
    # PointConv was renamed to PointNetConv in newer PyG versions
    PointConv = gn.PointNetConv


class LinearLN(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2) -> nn.Module:
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(out_features)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.relu(x)
        x = self.ln(x)
        x = self.dropout(x)
        return x


class PointConvNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=3, dropout=0.3, aggr="mean"):
        super().__init__()
        self.num_layers = num_layers
        self.dense = LinearLN(input_dim, hidden_dim, dropout)
        self.hidden_conv = nn.ModuleList(
            [PointConv(nn.Linear(hidden_dim * 2, hidden_dim), aggr=aggr) for _ in range(num_layers - 1)]
        )
        self.last_conv = PointConv(nn.Linear(hidden_dim * 2, hidden_dim), aggr=aggr)

    def latent(self, x):
        return self.dense(x)

    def forward(self, x, edge_index, retain_latent=False):
        x = self.dense(x)
        h = x
        for conv in self.hidden_conv:
            x = conv(x, h, edge_index)
            x = x.relu()
        x = self.last_conv(x, h, edge_index)
        if retain_latent:
            return x, h
        return x


class GraphConvNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=3, dropout=0.3, aggr="mean"):
        super().__init__()
        self.num_layers = num_layers
        self.dense = LinearLN(input_dim, hidden_dim, dropout)
        self.hidden_conv = nn.ModuleList(
            [gn.GraphConv(hidden_dim, hidden_dim, aggr=aggr) for _ in range(num_layers - 1)]
        )
        self.last_conv = gn.GraphConv(hidden_dim, hidden_dim, aggr=aggr)

    def latent(self, x):
        return self.dense(x)

    def forward(self, x, edge_index, retain_latent=False):
        h = x = self.dense(x)
        for conv in self.hidden_conv:
            x = conv(x, edge_index)
            x = x.relu()
        x = self.last_conv(x, edge_index)
        if retain_latent:
            return x, h
        return x


class GCNConvNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=3, dropout=0.3, aggr="mean"):
        super().__init__()
        self.num_layers = num_layers
        self.dense = LinearLN(input_dim, hidden_dim, dropout)
        self.hidden_conv = nn.ModuleList([gn.GCNConv(hidden_dim, hidden_dim, aggr=aggr) for _ in range(num_layers - 1)])
        self.last_conv = gn.GCNConv(hidden_dim, hidden_dim, aggr=aggr)

    def latent(self, x):
        return self.dense(x)

    def forward(self, x, edge_index, retain_latent=False):
        h = x = self.dense(x)
        for conv in self.hidden_conv:
            x = conv(x, edge_index)
            x = x.relu()
        x = self.last_conv(x, edge_index)
        if retain_latent:
            return x, h
        return x


class GATConvNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=3, dropout=0.3, aggr="mean"):
        super().__init__()
        self.num_layers = num_layers
        self.dense = LinearLN(input_dim, hidden_dim, dropout)
        num_heads = 8
        self.hidden_conv = nn.ModuleList(
            [gn.GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads) for _ in range(num_layers - 1)]
        )
        self.last_conv = gn.GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)

    def latent(self, x):
        return self.dense(x)

    def forward(self, x, edge_index, retain_latent=False):
        h = x = self.dense(x)
        for conv in self.hidden_conv:
            x = conv(x, edge_index)
            x = x.relu()
        x = self.last_conv(x, edge_index)
        if retain_latent:
            return x, h
        return x


def getBaseModel(name="GraphConv", *args, **kwargs) -> nn.Module:
    if name == "GraphConv":
        return GraphConvNet(*args, **kwargs)
    elif name == "PointConv":
        return PointConvNet(*args, **kwargs)
    elif name == "GCNConv":
        return GCNConvNet(*args, **kwargs)
    elif name == "GATConv":
        return GATConvNet(*args, **kwargs)
    else:
        raise KeyError
