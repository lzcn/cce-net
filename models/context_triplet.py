import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers as L


class ContextTripletNet(nn.Module):
    """
    Context Conditioning Embedding network trained with the triplet loss.

    Models:

       x -> self.encoder.latent -> self.encoder.conv -> context conditioned embedding
       x -> self.encoder.latent -> self.trans -> general embedding

    """

    def __init__(
        self,
        base_model="GATConv",
        input_dim=512,
        hidden_dim=128,
        dropout=0.2,
        num_layers=2,
        aggr="mean",
        norm=False,
        without_context=False,
        trans_layer=False,
        tau=0.1,
        use_critic_score=False,
        fuse_score=True,
        fuse_beta=0.5,
        learn_fuse_beta=False,
        loss_type="triplet",
    ):
        super().__init__()
        self.encoder = L.getBaseModel(
            name=base_model,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.without_context = without_context
        self.trans_layer = trans_layer
        self.tau = tau
        self.norm = norm
        self.dim = hidden_dim
        self.features = None
        # use density ratio instead of normalized compatibility score
        self.use_critic_score = use_critic_score
        self.fuse_score = fuse_score
        self.learn_fuse_beta = learn_fuse_beta
        self.loss_type = loss_type
        assert loss_type in ["infonce", "triplet"]
        # general embedding branch
        if trans_layer:
            self.trans = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        else:
            self.trans = nn.Identity()
        if self.learn_fuse_beta:
            self.beta = nn.parameter.Parameter(torch.zeros(1))
        else:
            self.beta = fuse_beta

    @property
    def fuse_beta(self):
        if self.learn_fuse_beta:
            return torch.sigmoid(self.beta)
        else:
            return self.beta

    @torch.no_grad()
    def latent(self, features: torch.Tensor, item_mask: torch.Tensor):
        """Build feature pool for all items."""
        h = self.gel_branch(features)
        self.features = h
        self.split_mask = item_mask.view(1, -1)

    def cce_branch(self, context, item_mask, edge_index):
        if self.without_context:
            # get x_i without context
            context = context[item_mask == 1]
            context = self.encoder.latent(context)
            h = self.trans(context)
        else:
            # get x_i with context
            context = self.encoder(context, edge_index)
            h = context[item_mask == 1]
        if self.norm:
            h = F.normalize(h, dim=-1)
        return h

    def gel_branch(self, x, item_mask=None):
        if item_mask is not None:
            x = x[item_mask == 1]
        x = self.encoder.latent(x)
        x = self.trans(x)
        if self.norm:
            x = F.normalize(x, dim=-1)
        return x

    @torch.no_grad()
    def eval_batch(self, context, target_index, edge_index, num_pairs, item_mask):
        if self.fuse_score:
            z_c = self.cce_branch(context, item_mask, edge_index)
            z_g = self.gel_branch(context, item_mask)
            z = self.fuse_beta * z_c + (1 - self.fuse_beta) * z_g
        else:
            z = self.cce_branch(context, item_mask, edge_index)
        assert len(z) == len(target_index) == num_pairs.sum().item()
        # n x num_items
        scores = (z * self.features[target_index.view(-1), :]).sum(dim=-1)
        compatibility = []
        offset = 0
        for s in num_pairs.tolist():
            compatibility.append(scores[offset : offset + s].mean().item())
            offset += s
        return compatibility

    def train_batch(self, x, pos, neg, edge_index, batch_size, item_mask):
        if self.fuse_score:
            z_c = self.cce_branch(x, item_mask, edge_index)
            z_g = self.gel_branch(x, item_mask)
            anc = self.fuse_beta * z_c + (1 - self.fuse_beta) * z_g
        else:
            anc = self.cce_branch(x, item_mask, edge_index)
        pos = self.gel_branch(pos)
        neg = self.gel_branch(neg)
        # b x 1 x dim
        anc = anc.reshape(batch_size, -1, self.dim)
        # b x v x dim
        neg = neg.reshape(batch_size, -1, self.dim)
        # b x 1 x dim
        pos = pos.reshape(batch_size, -1, self.dim)
        # b x 1 x 1
        pos_score = (anc * pos).sum(dim=-1)
        # b x v x 1
        neg_score = (anc * neg).sum(dim=-1)
        loss = F.relu(neg_score - pos_score + 0.1).mean()
        acc = (pos_score > neg_score) * 1.0
        return loss, acc.mean()

    def forward(self, *inputs):
        if self.training:
            return self.train_batch(*inputs)
        return self.eval_batch(*inputs)
