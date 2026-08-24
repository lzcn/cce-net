import logging
import os
import random

import lmdb
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from . import utils
from .generator import getGenerator

LOGGER = logging.getLogger(__name__)


def load_features(feature_dir: str, index: utils.ItemIndex) -> torch.Tensor:
    """Load all item features from an LMDB store into a single tensor.

    Rows follow the global node id order defined by ``index``.
    """
    env = lmdb.open(feature_dir, readonly=True, lock=False, readahead=False, meminit=False)
    features = torch.empty(index.num_items, 512, dtype=torch.float32)
    with env.begin(write=False) as txn:
        for cate, keys in enumerate(index.item_list):
            offset = int(index.offsets[cate])
            for local_id, key in enumerate(keys):
                buf = txn.get(key.encode())
                if buf is None:
                    raise KeyError(f"Feature for item {key} not found in {feature_dir}")
                features[offset + local_id] = torch.from_numpy(np.frombuffer(buf, dtype=np.float32).copy())
    env.close()
    return features


class ContexItemDataset(Dataset):
    """Outfit dataset that builds a context graph for each (anchor, target) pair.

    Data layout (see https://huggingface.co/datasets/lzcn/outfit-datasets):

        {root}/{dataset}/features/resnet34/data.mdb   # pre-extracted ResNet-34 features
        {root}/{dataset}/{split}/items.json           # item keys grouped by category
        {root}/{dataset}/{split}/{phase}_pos          # positive tuples
        {root}/{dataset}/{split}/{phase}_neg          # negative tuples (optional)

    Args:
        root (str): data root of the outfit datasets.
        data_set (str): relative path of the split, e.g. ``maryland-polyvore/original``.
        phase (str): one of ``train``, ``valid``, ``test``.
    """

    def __init__(
        self,
        root="../outfits-hf",
        data_set="maryland-polyvore/original",
        phase="train",
        normalize=False,
        num_neg=64,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        neg_mode="RandomMix",
        num_replace=1,
        neg_ratio=1,
        type_aware=True,
        full_graph=False,
    ):
        super().__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.phase = phase
        self.num_neg = num_neg
        self.training = phase == "train"
        self.full_graph = full_graph
        split_dir = os.path.join(self.root, data_set)
        dataset_dir = os.path.join(self.root, data_set.split("/")[0])
        # load item index and features
        self.index = utils.ItemIndex(utils.load_json(split_dir, "items.json"))
        LOGGER.info("Number of items: %d (%d categories)", self.index.num_items, self.index.num_types)
        self.features = load_features(os.path.join(dataset_dir, "features", "resnet34"), self.index)
        LOGGER.info("Shape of features: %s", tuple(self.features.shape))
        if normalize:
            LOGGER.info("Normalizing the features.")
            self.features = utils.normalize_features(self.features)
        self.node_type = {node: int(cate) for node, cate in enumerate(self.index.node_type)}
        self.num_types = self.index.num_types
        # get positive and negative outfits
        pos_fn = os.path.join(split_dir, f"{phase}_pos")
        neg_fn = os.path.join(split_dir, f"{phase}_neg")
        self.pos_tuples = np.array(utils.load_csv(pos_fn, converter=int))
        self.neg_tuples = None
        if os.path.exists(neg_fn):
            self.neg_tuples = np.array(utils.load_csv(neg_fn, converter=int))
        # generative negative tuples
        self.generator = getGenerator(
            mode=neg_mode, data=self.neg_tuples, ratio=neg_ratio, num_replace=num_replace, type_aware=type_aware
        )
        self.type_aware = type_aware
        # convert outfits by item id to outfits by node id
        outfit_list = utils.convert_tuple_to_node(self.pos_tuples, self.index)
        self.full_node_list = list(set([item for outfit in outfit_list for item in outfit]))
        self.type_node_list = [list() for _ in range(self.num_types)]
        for node in self.full_node_list:
            self.type_node_list[self.node_type[node]].append(node)
        # item mask for current split
        item_mask = torch.zeros(len(self.features))
        item_mask[self.full_node_list] = 1
        self.item_mask = item_mask > 0
        assert self.item_mask.sum().item() == len(self.full_node_list)
        LOGGER.info("Number of items in %s set: %d", self.phase, len(self.full_node_list))
        self.mode = "train" if self.training else "eval"
        self.build()

    def build(self):
        self.neg_tuples = self.generator(self.pos_tuples)
        # converted into node format
        self.pos = utils.convert_tuple_to_node(self.pos_tuples, self.index)
        self.neg = utils.convert_tuple_to_node(self.neg_tuples, self.index)
        # for testing
        self.outfits = self.pos + self.neg
        self.labels = [1] * len(self.pos) + [0] * len(self.neg)
        self.suggestions = self.pos
        if self.mode == "train":
            self.datasize = len(self.pos)
        elif self.mode == "eval":
            self.datasize = len(self.outfits)
        else:
            self.datasize = len(self.suggestions)

    def use_suggestion(self):
        self.mode = "suggestion"

    def get_suggestion(self, index):
        outfit = self.suggestions[index].copy()
        index = np.random.choice(len(outfit))
        node = outfit[index]
        node_type = self.node_type[node]
        outfit.pop(index)
        negative = random.choice(self.type_node_list[node_type])
        while negative == node:
            negative = random.choice(self.type_node_list[node_type])
        suggestion = [negative] + list(outfit)
        num_items = len(suggestion)
        indx = []
        indy = []
        for i in range(num_items):
            for j in range(num_items):
                if i != j:
                    indx.append(i)
                    indy.append(j)
        item_list = []
        edge_index = []
        offset = 0
        item_mask = []
        query_idx = []
        num_graphs = 0
        for x, y in zip(indx, indy):
            # score "query -> target"
            query = suggestion[x]
            target = suggestion[y]
            context = set(suggestion.copy())
            context.discard(target)
            context.discard(query)
            # move anchor to the first place
            graph = [target] + list(context)
            mask = [1] + [0] * len(context)
            num_nodes = len(graph)
            item_list += graph
            item_mask += mask
            query_idx.append(query)
            if self.full_graph:
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        edge_index.append((i + offset, j + offset))
                        edge_index.append((j + offset, i + offset))
            else:
                if num_nodes == 1:
                    edge_index.append((offset, offset))
                for i in range(1, num_nodes):
                    edge_index.append((i + offset, 0 + offset))
            offset += num_nodes
            num_graphs += 1
        assert num_graphs == len(query_idx)
        data = Data(
            x=self.features[item_list],
            target=torch.LongTensor(query_idx),
            edge_index=torch.LongTensor(edge_index).t().contiguous(),
            num_nodes=len(item_list),
            y=self.labels[index],
            item_mask=torch.LongTensor(item_mask),
            num_pairs=num_graphs,
            num_items=num_items,
        )
        return data

    def get_eval(self, index):
        outfit = self.outfits[index]
        # remove duplicate items
        outfit = list(set(outfit))
        # all ordered pairs (i, j), i != j
        num_items = len(outfit)
        indx = []
        indy = []
        for i in range(num_items):
            for j in range(num_items):
                if i != j:
                    indx.append(i)
                    indy.append(j)
        item_list = []
        edge_index = []
        offset = 0
        item_mask = []
        target_idx = []
        num_graphs = 0
        for x, y in zip(indx, indy):
            anchor = outfit[x]
            target = outfit[y]
            context = set(outfit)
            context.discard(anchor)
            context.discard(target)
            # move anchor to the first place
            graph = [anchor] + list(context)
            mask = [1] + [0] * len(context)
            num_nodes = len(graph)
            item_list += graph
            item_mask += mask
            target_idx.append(target)
            if self.full_graph:
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        edge_index.append((i + offset, j + offset))
                        edge_index.append((j + offset, i + offset))
            else:
                for i in range(1, num_nodes):
                    edge_index.append((i + offset, 0 + offset))
            offset += num_nodes
            num_graphs += 1
        assert num_graphs == len(target_idx)
        data = Data(
            x=self.features[item_list],
            target=torch.LongTensor(target_idx),
            edge_index=torch.LongTensor(edge_index).t().contiguous(),
            num_nodes=len(item_list),
            y=self.labels[index],
            item_mask=torch.LongTensor(item_mask),
            num_pairs=num_graphs,
            num_items=num_items,
        )
        return data

    def get_train(self, index):
        # get the positive outfit
        outfit = list(set(self.pos[index]))
        anc_node, pos_node = np.random.choice(outfit, 2, replace=False)
        # get the item pool
        if self.type_aware:
            node_type = self.node_type[pos_node]
            node_pool = self.type_node_list[node_type]
        else:
            node_pool = self.full_node_list
        # sample neg nodes
        neg_nodes = []
        for node in np.random.choice(node_pool, 2 * self.num_neg, replace=False):
            if node == anc_node or node == pos_node:
                continue
            neg_nodes.append(node)
        pos_nodes = [pos_node]
        neg_nodes = neg_nodes[: self.num_neg]
        # build graph
        context = set(outfit)
        context.discard(anc_node)
        context.discard(pos_node)
        item_nodes = [anc_node] + list(context)
        item_masks = [1] + [0] * len(context)
        num_nodes = len(item_nodes)
        edge_index = []
        if self.full_graph:
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    edge_index.append((i, j))
                    edge_index.append((j, i))
        else:
            for i in range(1, num_nodes):
                edge_index.append((i, 0))

        x = self.features[item_nodes]
        data = Data(
            x=x,
            edge_index=torch.LongTensor(edge_index).t().contiguous(),
            num_nodes=num_nodes,
            pos=self.features[pos_nodes],
            neg=self.features[neg_nodes],
            item_mask=torch.LongTensor(item_masks),
        )
        return data

    def get(self, index):
        if self.mode == "train":
            return self.get_train(index)
        elif self.mode == "eval":
            return self.get_eval(index)
        else:
            return self.get_suggestion(index)

    def len(self):
        return self.datasize
