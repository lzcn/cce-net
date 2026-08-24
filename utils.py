"""Standalone utilities for logging, checkpointing and model IO.

Minimal replacements for the parts of `torchutils` used by this project.
"""

import logging
import os
import sys

import torch
from torch import nn


def config(logger_name="main", log_file=None, level=logging.INFO):
    """Configure the root logger with console and optional file output."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m/%d %H:%M:%S")
    if not logger.handlers:
        stream = logging.StreamHandler(stream=sys.stdout)
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def format_display(args: dict) -> str:
    """One-line representation of an argument dict."""
    return " ".join([f"{key}: {value}" for key, value in args.items()])


class ModelSaver:
    """Save model checkpoints.

    The latest model is saved to ``{prefix}_latest.pt`` and the best model
    (w.r.t. ``score``) is saved to ``{prefix}_best.pt``.

    Args:
        dirname (str): directory for checkpoints.
        filename_prefix (str): prefix of checkpoint files.
        save_best (bool): keep a copy of the best model. Defaults to ``False``.
        save_latest (bool): keep a copy of the latest model. Defaults to ``False``.
        mode (str): ``max`` or ``min``, whether a higher score is better.
    """

    def __init__(
        self,
        dirname: str,
        filename_prefix: str = "net",
        score_name: str = "score",
        save_best: bool = False,
        save_latest: bool = False,
        mode: str = "max",
    ):
        self.dirname = dirname
        self.filename_prefix = filename_prefix
        self.score_name = score_name
        self.save_best = save_best
        self.save_latest = save_latest
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self._latest_checkpoint = None
        os.makedirs(dirname, exist_ok=True)

    def _save(self, net: nn.Module, score, epoch, path):
        state = {"state_dict": net.state_dict(), "epoch": epoch, self.score_name: score}
        torch.save(state, path)

    def save(self, net: nn.Module, score, epoch) -> bool:
        """Save checkpoints, returns True if this is a new best model."""
        is_best = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        if self.save_latest:
            path = os.path.join(self.dirname, f"{self.filename_prefix}_latest.pt")
            self._save(net, score, epoch, path)
            self._latest_checkpoint = path
        if self.save_best and is_best:
            self.best_score = score
            path = os.path.join(self.dirname, f"{self.filename_prefix}_best.pt")
            self._save(net, score, epoch, path)
            self._latest_checkpoint = path
        return is_best

    @property
    def last_checkpoint(self):
        assert self._latest_checkpoint is not None, "No checkpoint has been saved yet."
        return self._latest_checkpoint


def load_pretrained(net: nn.Module, path_or_state_dict) -> nn.Module:
    """Load matching weights from a checkpoint file or state dict."""
    if isinstance(path_or_state_dict, str):
        state = torch.load(path_or_state_dict, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    else:
        state = path_or_state_dict
    model_state = net.state_dict()
    matched = {name: param for name, param in state.items() if name in model_state and model_state[name].shape == param.shape}
    unmatched = [name for name in state.keys() if name not in matched]
    missing = [name for name in model_state.keys() if name not in matched]
    net.load_state_dict(matched, strict=False)
    logging.getLogger(__name__).info(
        "Loaded %d weights, skipped %d unmatched, %d missing.", len(matched), len(unmatched), len(missing)
    )
    return net
