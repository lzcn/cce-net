import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers as L


class ContextContrastiveNet(nn.Module):
    """
    Context Conditioning Embedding network.

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
        fuse_isomeric=False,
        learn_fuse_beta=False,
        loss_type="infonce",
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
        self.trans_layer = trans_layer
        self.without_context = without_context
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
        self.features = h.transpose(0, 1)
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

    def forward_branch(self, context, item_mask, edge_index):
        if self.fuse_score:
            z_c = self.cce_branch(context, item_mask, edge_index)
            z_g = self.gel_branch(context, item_mask)
            z = self.fuse_beta * z_c + (1 - self.fuse_beta) * z_g
        else:
            z = self.cce_branch(context, item_mask, edge_index)
        return z

    @torch.no_grad()
    def eval_batch(self, context, target_index, edge_index, num_items, item_mask):
        z = self.forward_branch(context, item_mask, edge_index)
        n = len(z)
        num_pairs = (num_items - 1) * num_items
        assert len(z) == len(target_index) == num_pairs.sum().item()
        # n x num_items
        if self.use_critic_score:
            # use unnormalized density ratio the compatibility score
            # the compatibility is independent of tau
            logprob = (z * self.features.transpose(1, 0)[target_index.view(-1), :]).sum(dim=-1) / self.tau
        else:
            scores = z.matmul(self.features) / self.tau
            # mask out items in other splits
            inf = torch.finfo(scores.dtype).max
            prob = F.softmax(scores.masked_fill_(~self.split_mask.bool(), -torch.inf), dim=-1)
            logprob = torch.log(prob[torch.arange(n), target_index.view(-1)])
        comparibility = []
        offset = 0
        for n_item in num_items.tolist():
            n_pair = n_item * (n_item - 1)
            scores = logprob[offset : offset + n_pair].reshape(n_item, n_item - 1)
            scores = scores.mean(dim=-1)
            comparibility.append(torch.mean(scores).item())
            offset += n_pair
        assert offset == n
        return comparibility

    @torch.no_grad()
    def eval_contribution(self, context, target_index, edge_index, num_items, item_mask):
        z = self.forward_branch(context, item_mask, edge_index)
        n = len(z)
        num_pairs = (num_items - 1) * num_items
        assert len(z) == len(target_index) == num_pairs.sum().item()
        scores = z.matmul(self.features) / self.tau
        # mask out items in other splits
        inf = torch.finfo(scores.dtype).max
        prob = F.softmax(scores.masked_fill_(~self.split_mask.bool(), -torch.inf), dim=-1)
        # n
        logprob = torch.log(prob[torch.arange(n), target_index.view(-1)])
        index = []
        offset = 0
        for n_item in num_items.tolist():
            n_pair = n_item * (n_item - 1)
            scores = logprob[offset : offset + n_pair].reshape(n_item, n_item - 1)
            scores = scores.mean(dim=-1)
            index.append(torch.argmin(scores).item())
            offset += n_pair
        assert offset == n
        return index

    def train_batch(self, x, pos, neg, edge_index, batch_size, item_mask):
        anc = self.forward_branch(x, item_mask, edge_index)
        pos = self.gel_branch(pos)
        neg = self.gel_branch(neg)
        # b x 1 x d
        anc = anc.reshape(batch_size, -1, self.dim)
        # b x 1 x d
        pos = pos.reshape(batch_size, -1, self.dim)
        # b x v x d
        neg = neg.reshape(batch_size, -1, self.dim)
        if self.loss_type == "infonce":
            loss, accuracy = self.info_nce_loss(anc, pos, neg)
        else:
            loss, accuracy = self.triplet_loss(anc, pos, neg)
        return loss, accuracy

    def info_nce_loss(self, anc, pos, neg):
        # b x (1 + v) x d
        target = torch.cat((pos, neg), dim=1).transpose(1, 2)
        # b x (1 + v)
        critic = anc.matmul(target).squeeze(1) / self.tau
        accuracy = (torch.argmax(critic, dim=-1) == 0).sum() / len(critic)
        logprob = -F.log_softmax(critic, dim=-1)
        loss = logprob[:, 0].mean()
        return loss, accuracy

    def triplet_loss(self, anc, pos, neg):
        # b x 1 x 1
        pos_score = (anc * pos).sum(dim=-1)
        # b x v x 1
        neg_score = (anc * neg).sum(dim=-1)
        loss = F.relu(neg_score - pos_score + 0.1).mean()
        acc = (pos_score > neg_score) * 1.0
        acc = acc.mean()
        return loss, acc

    def forward(self, *inputs):
        if self.training:
            return self.train_batch(*inputs)
        return self.eval_batch(*inputs)


class HybridContextContrastiveNet(nn.Module):
    """
    Hybrid variant of CCE with two separate critics.

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
        fuse_isomeric=False,
        learn_fuse_beta=False,
        loss_type="infonce",
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
        self.trans_layer = trans_layer
        self.without_context = without_context
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
        self.features = h.transpose(0, 1)
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

    def forward_branch(self, context, item_mask, edge_index):
        z_c = self.cce_branch(context, item_mask, edge_index)
        z_g = self.gel_branch(context, item_mask)
        return z_c, z_g

    @torch.no_grad()
    def eval_batch(self, context, target_index, edge_index, num_items, item_mask):
        z_c, z_g = self.forward_branch(context, item_mask, edge_index)
        n = len(z_c)
        num_pairs = (num_items - 1) * num_items
        assert len(z_c) == len(target_index) == num_pairs.sum().item()
        # n x num_items
        if self.use_critic_score:
            # use unnormalized density ratio the compatibility score
            # the compatibility is independent of tau
            logprob = (z_c * self.features.transpose(1, 0)[target_index.view(-1), :]).sum(dim=-1) / self.tau
        else:
            score_c = z_c.matmul(self.features) / self.tau
            score_g = z_g.matmul(self.features) / self.tau
            # mask out items in other splits
            inf = torch.finfo(score_c.dtype).max
            prob_c = F.softmax(score_c.masked_fill_(~self.split_mask.bool(), -torch.inf), dim=-1)
            prob_g = F.softmax(score_g.masked_fill_(~self.split_mask.bool(), -torch.inf), dim=-1)
            logprob_c = torch.log(prob_c[torch.arange(n), target_index.view(-1)])
            logprob_g = torch.log(prob_g[torch.arange(n), target_index.view(-1)])
            logprob = self.fuse_beta * logprob_c + (1 - self.fuse_beta) * logprob_g
        comparibility = []
        offset = 0
        for n_item in num_items.tolist():
            n_pair = n_item * (n_item - 1)
            score = logprob[offset : offset + n_pair].reshape(n_item, n_item - 1)
            score = score.mean(dim=-1)
            comparibility.append(torch.mean(score).item())
            offset += n_pair
        assert offset == n
        return comparibility

    @torch.no_grad()
    def eval_contribution(self, context, target_index, edge_index, num_items, item_mask):
        z_c, z_g = self.forward_branch(context, item_mask, edge_index)
        z = self.fuse_beta * z_c + (1 - self.fuse_beta) * z_g
        n = len(z)
        num_pairs = (num_items - 1) * num_items
        assert len(z) == len(target_index) == num_pairs.sum().item()
        scores = z.matmul(self.features) / self.tau
        # mask out items in other splits
        inf = torch.finfo(scores.dtype).max
        prob = F.softmax(scores.masked_fill_(~self.split_mask.bool(), -torch.inf), dim=-1)
        # n
        logprob = torch.log(prob[torch.arange(n), target_index.view(-1)])
        index = []
        offset = 0
        for n_item in num_items.tolist():
            n_pair = n_item * (n_item - 1)
            scores = logprob[offset : offset + n_pair].reshape(n_item, n_item - 1)
            scores = scores.mean(dim=-1)
            index.append(torch.argmin(scores).item())
            offset += n_pair
        assert offset == n
        return index

    def train_batch(self, x, pos, neg, edge_index, batch_size, item_mask):
        anc_l, anc_g = self.forward_branch(x, item_mask, edge_index)
        pos = self.gel_branch(pos)
        neg = self.gel_branch(neg)
        # b x 1 x d
        anc_l = anc_l.reshape(batch_size, -1, self.dim)
        anc_g = anc_g.reshape(batch_size, -1, self.dim)
        # b x 1 x d
        pos = pos.reshape(batch_size, -1, self.dim)
        # b x v x d
        neg = neg.reshape(batch_size, -1, self.dim)
        # b x (1 + v) x d
        target = torch.cat((pos, neg), dim=1).transpose(1, 2)
        # b x (1 + v)
        critic_l = anc_l.matmul(target).squeeze(1) / self.tau
        critic_g = anc_g.matmul(target).squeeze(1) / self.tau
        logprob_l = -F.log_softmax(critic_l, dim=-1)
        logprob_g = -F.log_softmax(critic_g, dim=-1)
        critic = self.fuse_beta * logprob_l + (1 - self.fuse_beta) * logprob_g
        accuracy = (torch.argmax(critic, dim=-1) == 0).sum() / len(critic)
        loss = self.fuse_beta * logprob_l[:, 0].mean() + (1 - self.fuse_beta) * logprob_g[:, 0].mean()
        return loss, accuracy

    def forward(self, *inputs):
        if self.training:
            return self.train_batch(*inputs)
        return self.eval_batch(*inputs)
