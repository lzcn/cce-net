import csv
import json
import logging
import os
from typing import List

import numpy as np
import torch

from .generator import split_tuple

LOGGER = logging.getLogger(__name__)


def load_json(fn, *args):
    fn = os.path.join(fn, *args)
    with open(fn) as f:
        data = json.load(f)
    return data


def load_csv(fn, converter=None):
    """Load data in csv format."""
    fn = os.path.expanduser(fn)
    with open(fn, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = list(reader)
        if converter is not None:
            data = [list(map(converter, line)) for line in data]
    return data


@torch.no_grad()
def normalize_features(feats, mean=None, std=None):
    mean = feats.mean(dim=0) if mean is None else mean
    std = feats.std(dim=0) if std is None else std
    feats = (feats - mean) / std
    return feats


class ItemIndex:
    """Maps (category, local item index) pairs to global node ids.

    Items are grouped by category following ``items.json`` from the outfit datasets,
    i.e. ``item_list[c][i]`` is the feature key of the i-th item in category ``c``.
    The node id of an item is its position in the flattened list.
    """

    def __init__(self, item_list: List[List[str]]):
        self.item_list = item_list
        self.num_types = len(item_list)
        sizes = np.array([len(g) for g in item_list])
        self.offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        self.num_items = int(sizes.sum())
        # node id -> category
        self.node_type = np.repeat(np.arange(self.num_types), sizes)

    def to_node(self, item: int, cate: int) -> int:
        return int(self.offsets[cate] + item)

    def key(self, item: int, cate: int) -> str:
        return self.item_list[cate][item]


def convert_tuple_to_node(tuples: np.ndarray, index: ItemIndex) -> List[List[int]]:
    """Convert tuples into lists of unique node ids, dropping padded slots."""
    _, sizes, item_ids, item_types = split_tuple(tuples)
    outfits = []
    for n, items, types in zip(sizes, item_ids, item_types):
        nodes = list({index.to_node(i, c) for i, c in zip(items[:n], types[:n])})
        if len(nodes) < 2:
            LOGGER.warning("Skipping bad outfit with %d unique items: %s", len(nodes), nodes)
        else:
            outfits.append(nodes)
    return outfits
