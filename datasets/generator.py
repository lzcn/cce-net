"""Generators for negative outfit tuples.

Tuples are stored as ``np.ndarray`` of shape ``(n, 2 + 2 * max_size)`` where each
row is ``[uidx, size, item_1, ..., item_K, type_1, ..., type_K]`` padded with -1.
``item_i`` is the index of the item within its category and ``type_i`` is the
category index.

This is a trimmed-down port of the generators in [lzcn/outfit-datasets](https://github.com/lzcn/outfit-datasets).
"""

import logging
from typing import Any, Callable, List

import numpy as np

NONE_TYPE = -1

_generator_registry = {}

LOGGER = logging.getLogger(__name__)


def split_tuple(tuples: np.ndarray) -> List[np.ndarray]:
    """Split tuples into (uids, sizes, item_ids, item_types)."""
    uids = tuples[:, 0]
    length = tuples[:, 1]
    item_ids, item_types = np.split(tuples[:, 2:], 2, axis=1)
    return uids, length, item_ids, item_types


def infer_max_shape(tuples: np.ndarray) -> int:
    """Max number of item slots per tuple."""
    return (tuples.shape[1] - 2) // 2


def infer_num_type(tuples: np.ndarray) -> int:
    """Infer the number of categories from tuples."""
    item_types = set(split_tuple(tuples)[-1].flatten())
    if NONE_TYPE in item_types:
        num_type = len(item_types) - 1
    else:
        num_type = len(item_types)
    return num_type


def get_item_list(data: np.ndarray) -> List[np.ndarray]:
    """Return the item pool for each category."""
    _, _, item_ids, item_types = split_tuple(data)
    num_list = len(set(item_types.flatten()))
    item_set = [set() for _ in range(num_list)]
    for idxs, types in zip(item_ids, item_types):
        for idx, c in zip(idxs, types):
            item_set[c].add(idx)
    return [np.array(list(s)) for s in item_set]


class Generator:
    """Base class for tuple generators."""

    run: Callable[..., Any] = None

    def __init_subclass__(cls):
        super().__init_subclass__()
        _generator_registry[cls.__name__] = cls

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def __call__(self, data: np.ndarray = None) -> np.ndarray:
        self.logger.info("Generating tuples with %s mode.", self)
        return self.run(data)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.extra_repr() + ")"


class Fix(Generator):
    """Always return the saved tuples."""

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__()
        assert data is not None, "Fix mode requires pre-generated negative tuples."
        self.data = data

    def run(self, *input: Any) -> np.ndarray:
        return self.data


class RandomMix(Generator):
    """Generate negative outfits by randomly mixing items.

    Given an outfit :math:`{x_1, ..., x_n}`, sample a new item :math:`x_i^-` for
    each slot to get a negative outfit :math:`{x_1^-, ..., x_n^-}`.

    Args:

        ratio (int): ratio of negative outfits to be sampled for each positive outfit.
        type_aware (bool): whether to keep the category of each item when sampling.

    """

    def __init__(self, ratio: int = 1, type_aware: bool = False, **kwargs):
        super().__init__()
        self.type_aware = type_aware
        self.ratio = ratio

    def run(self, data: np.ndarray) -> np.ndarray:
        item_list = get_item_list(data)
        num_items = list(map(len, item_list))
        num_types = infer_num_type(data)
        max_items = infer_max_shape(data)
        pos_uids, pos_sizes, pos_items, pos_types = split_tuple(data)
        neg_uids = pos_uids.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_sizes = pos_sizes.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_types = []
        neg_items = []
        pos_set = set(map(tuple, pos_items))
        for size, item_types in zip(pos_sizes, pos_types):
            n_sampled = 0
            while n_sampled < self.ratio:
                if self.type_aware:
                    sampled_types = item_types
                else:
                    sampled_types = np.random.randint(num_types, size=max_items)
                sampled_items = [np.random.choice(item_list[i]) for i in sampled_types]
                sampled_items = sampled_items[:size] + [NONE_TYPE] * (max_items - size)
                if tuple(sampled_items) not in pos_set:
                    n_sampled += 1
                    neg_items.append(sampled_items)
                    neg_types.append(sampled_types)
        neg_items = np.array(neg_items)
        neg_types = np.array(neg_types)
        return np.hstack([neg_uids, neg_sizes, neg_items, neg_types])

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}, type_aware={self.type_aware}"


class RandomReplace(Generator):
    """Generate negative outfits by replacing items in positive outfits.

    Args:

        ratio (int): ratio of negative outfits per positive outfit.
        num_replace (int): number of items to replace in each outfit.
        type_aware (bool): whether to sample replacements from the same category.

    """

    def __init__(self, ratio=1, num_replace=1, type_aware=False, **kwargs):
        super().__init__()
        self.ratio = ratio
        self.num_replace = num_replace
        self.type_aware = type_aware

    def run(self, data: np.ndarray) -> np.ndarray:
        data = data.copy().repeat(self.ratio, axis=0)
        uids, pos_sizes, pos_items, pos_types = split_tuple(data)
        item_list = get_item_list(data)
        num_types = infer_num_type(data)
        pos_set = set(map(tuple, pos_items))
        neg_items = []
        neg_types = []
        for size, items, types in zip(pos_sizes, pos_items, pos_types):
            num_replace = min(size, self.num_replace)
            replace_index = np.random.choice(size, num_replace, replace=False)
            while tuple(items) in pos_set:
                for idx in replace_index:
                    target_type = types[idx]
                    target_item = items[idx]
                    # random sample an item
                    sampled_item = target_item
                    while sampled_item == target_item:
                        if self.type_aware:
                            sampled_type = target_type
                            sampled_item = np.random.choice(item_list[target_type])
                        else:
                            sampled_type = np.random.randint(num_types)
                            sampled_item = np.random.choice(item_list[sampled_type])
                    # replace item and type
                    items[idx] = sampled_item
                    types[idx] = sampled_type
            neg_items.append(items)
            neg_types.append(types)
        neg_items = np.array(neg_items)
        neg_types = np.array(neg_types)
        neg_data = np.hstack((uids.reshape((-1, 1)), pos_sizes.reshape((-1, 1)), neg_items, neg_types))
        return neg_data

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}, num_replace={self.num_replace}, type_aware={self.type_aware}"


def getGenerator(mode: str = None, data=None, **kwargs) -> Generator:
    """Get a tuple generator by name."""
    if mode is None:
        return None
    supported_modes = ",".join(_generator_registry.keys())
    assert mode in _generator_registry, f"Generator mode {mode} is not support. Only {supported_modes} are supported."
    return _generator_registry[mode](data=data, **kwargs)
