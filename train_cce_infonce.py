#!/usr/bin/env python
import argparse
import logging
import os
import textwrap

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

# fmt: off
parser = argparse.ArgumentParser(
    prog="CCE-Net", formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
    Learning Fashion Compatibility with Context Conditioning Embedding.
    Our model consists of two branches:
       1. A GCN-based context conditional embedding encoder
           x (node) -> LinearLN -> GCNs -> z (latent)
       2. A MLP-based general embedding encoder
           x (node) -> LinearLN -> MLPs -> z (latent)

    - LinearLN: feature extraction
    - GCNs: context injection.
    - MLPs: general embedding.

    '''))
# data parameters
parser.add_argument("--data-dir", default="../outfits-hf", help="Root of the outfit datasets")
parser.add_argument("--data-set", default="maryland-polyvore/original", help="Relative path of the data split")
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--neg-mode", default="RandomMix")
parser.add_argument("--num-replace", default=1, type=int, help="For RandomReplace mode. Number of negative samples to replace")
parser.add_argument("--neg-ratio", default=10, type=int)
parser.add_argument("--neg-type-aware", action="store_true", help="Sample negative samples from the same category")
# model parameters
parser.add_argument("--base-model", default="GATConv", help="Name of base model")
parser.add_argument("--full-graph", action="store_true", help="Symmetric context injection for all nodes. Otherwise, context only flow into target node")
parser.add_argument("--aggr", default="mean", help="aggregration for graph")
parser.add_argument("--num-layers", default=2, type=int, help="Number of GATConv layers")
parser.add_argument("--feat-norm", action="store_true", help="Normalize embedding x")
parser.add_argument("--norm", action="store_true", help="Normalize embedding z")
parser.add_argument("--tau", default=0.1, type=float, help="Scale the similarity (inner product)")
parser.add_argument("--without-context", action="store_true", help="Delete the context conditional branch")
parser.add_argument("--trans-layer", action="store_true", help="Add MLPs in general embedding branch")
parser.add_argument("--num-neg", default=64, type=int, help="Number of negative items for InfoNCE loss")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout for LinearLN")
parser.add_argument("--fuse-score", action="store_true", help="Fuse general embedding and context conditional embedding")
parser.add_argument("--fuse-isomeric", action="store_true", help="Fuse which general embedding to context conditional embedding")
parser.add_argument("--fuse-beta", default=0.5, type=float, help="beta * z_i_c + (1 - beta) * z_i")
parser.add_argument("--learn-fuse-beta", action="store_true", help="Learn beta")
parser.add_argument("--loss-type", default="InfoNCE", help="Triplet loss or InfoNCE loss")
parser.add_argument("--load-trained", default=None, type=str, help="Load pre-trained model")
parser.add_argument("--use-critic-score", action="store_true", help="Use density ratio as the score")
# optimizer parameters
parser.add_argument("--num-epochs", default=200, type=int)
parser.add_argument("--adam", action="store_true")
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--betas", default="0.9,0.999")
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight-decay", default=1e-6, type=float)
# script parameters
parser.add_argument("--device", default=0, type=int, help="CUDA device to use")
parser.add_argument("--num-runs", default=10, type=int, help="Number of runs")
parser.add_argument("--test", action="store_true")
parser.add_argument("--test-suggestion", action="store_true")
parser.add_argument("--log-dir", default="runs/exp", help="Name of exmeriments")
parser.add_argument("--log-name", default=None, type=str, help="Name of logfile")
# fmt: on

logger = logging.getLogger("main")
args = parser.parse_args()
args.betas = tuple(map(float, args.betas.split(",")))

os.makedirs(f"{args.log_dir}", exist_ok=True)
os.makedirs(f"{args.log_dir}/checkpoints", exist_ok=True)

# logfile
if args.log_name is None:
    name = "test" if args.test else "train"
else:
    name = args.log_name
log_file = f"{args.log_dir}/{name}.log"

utils.config(log_file=log_file)
logger.info(utils.format_display(vars(args)))

train_data = datasets.ContexItemDataset(
    root=args.data_dir,
    data_set=args.data_set,
    phase="train",
    num_neg=args.num_neg,
    normalize=args.feat_norm,
    full_graph=args.full_graph,
    neg_mode=args.neg_mode,
    neg_ratio=1,
    num_replace=args.num_replace,
    type_aware=args.neg_type_aware,
)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

valid_data = datasets.ContexItemDataset(
    root=args.data_dir,
    data_set=args.data_set,
    phase="valid",
    num_neg=args.num_neg,
    normalize=args.feat_norm,
    full_graph=args.full_graph,
    neg_mode=args.neg_mode,
    neg_ratio=args.neg_ratio,
    num_replace=args.num_replace,
    type_aware=args.neg_type_aware,
)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

test_data = datasets.ContexItemDataset(
    root=args.data_dir,
    data_set=args.data_set,
    phase="test",
    num_neg=args.num_neg,
    normalize=args.feat_norm,
    full_graph=args.full_graph,
    neg_mode=args.neg_mode,
    neg_ratio=args.neg_ratio,
    num_replace=args.num_replace,
    type_aware=args.neg_type_aware,
)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

if args.fuse_isomeric:
    net = models.HybridContextContrastiveNet(
        base_model=args.base_model,
        input_dim=512,
        hidden_dim=128,
        num_layers=args.num_layers,
        dropout=args.dropout,
        norm=args.norm,
        without_context=args.without_context,
        trans_layer=args.trans_layer,
        aggr=args.aggr,
        tau=args.tau,
        use_critic_score=args.use_critic_score,
        fuse_score=args.fuse_score,
        fuse_beta=args.fuse_beta,
        learn_fuse_beta=args.learn_fuse_beta,
        loss_type=args.loss_type.lower(),
    )
else:
    net = models.ContextContrastiveNet(
        base_model=args.base_model,
        input_dim=512,
        hidden_dim=128,
        num_layers=args.num_layers,
        dropout=args.dropout,
        norm=args.norm,
        without_context=args.without_context,
        trans_layer=args.trans_layer,
        aggr=args.aggr,
        tau=args.tau,
        use_critic_score=args.use_critic_score,
        fuse_score=args.fuse_score,
        fuse_beta=args.fuse_beta,
        learn_fuse_beta=args.learn_fuse_beta,
        loss_type=args.loss_type.lower(),
    )
net = net.cuda(args.device)


@torch.no_grad()
def eval(net: models.ContextContrastiveNet, loader: DataLoader):
    loader.dataset.build()
    net.eval()
    net.latent(loader.dataset.features.cuda(args.device), loader.dataset.item_mask.cuda(args.device))
    y_true = []
    y_pred = []
    for data in tqdm(loader, desc="Evaluating Recommendation AUC"):
        data = data.cuda(args.device)
        scores = net(data.x, data.target, data.edge_index, data.num_items, data.item_mask)
        y_pred += scores
        y_true += data.y.tolist()
    auc = roc_auc_score(y_true, y_pred)
    return auc


@torch.no_grad()
def eval_suggestion(net: models.ContextContrastiveNet, loader: DataLoader):
    loader.dataset.build()
    net.latent(loader.dataset.features.cuda(args.device), loader.dataset.item_mask.cuda(args.device))
    index_pred = []
    for data in tqdm(loader, desc="Evaluating Item Suggestion Acc"):
        data = data.cuda(args.device)
        index = net.eval_contribution(data.x, data.target, data.edge_index, data.num_items, data.item_mask)
        index_pred += index
    index_pred = np.array(index_pred)
    acc = (index_pred == 0).sum() / len(index_pred)
    print(acc)
    return acc.item()


if args.load_trained:
    utils.load_pretrained(net, args.load_trained)

if args.test:
    net = net.cuda(args.device)
    net = net.eval()
    aucs = []
    for n in range(args.num_runs):
        auc = eval(net, test_loader)
        aucs.append(auc)
        logger.info("[{}]/[{}] Run AUC: {:.4f}".format(n + 1, args.num_runs, auc))
    logger.info("Test AUC: {:.4f} +- {:.4f}".format(np.mean(aucs), np.std(aucs)))
    exit(0)

if args.test_suggestion:
    net = net.cuda()
    net = net.eval()
    test_data.use_suggestion()
    acc = np.array([eval_suggestion(net, test_loader) for _ in range(10)])
    test_auc_mean = np.mean(acc)
    test_auc_std = np.std(acc)
    logger.info("Test Accuracy: {:.4f} +- {:.4f}".format(test_auc_mean, test_auc_std))
    exit(0)


# checkpoints
saver = utils.ModelSaver(
    dirname=f"{args.log_dir}/checkpoints", filename_prefix="net", score_name="auc", save_best=True, save_latest=True
)

if args.adam:
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=args.betas)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5)
else:
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1)

# start training and evaluation
num_epochs = args.num_epochs
writer = SummaryWriter(args.log_dir)
best_auc = 0.0


test_auc = eval(net, test_loader)
logger.info("Test AUC: {:.4f}".format(test_auc))

logger.info("Starting training")
iterations = 0
for epoch in range(num_epochs):
    net.train()
    num_iters = len(train_loader)
    if net.fuse_score:
        with torch.no_grad():
            logger.info("Fuse Beta: {}".format(net.fuse_beta))
    for inter_count, data in enumerate(train_loader):
        data = data.cuda(args.device)
        optimizer.zero_grad()
        loss, acc = net(data.x, data.pos, data.neg, data.edge_index, args.batch_size, data.item_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
        if iterations % 10 == 0:
            logger.info(
                "Epoch: %d, Iter: %d / %d, Loss: %.3f, Accuracy: %.3f",
                epoch,
                inter_count,
                num_iters,
                loss.item(),
                acc.item(),
            )
        writer.add_scalar("Train/Loss", loss, iterations)
        writer.add_scalar("Train/Acc", acc, iterations)
        iterations += 1
        inter_count += 1
    # evaluate on the validation set
    val_auc = eval(net, valid_loader)
    if epoch % 20 == 0:
        test_auc = eval(net, test_loader)
    lr_scheduler.step(val_auc)
    # save files
    best_auc = max(best_auc, val_auc)
    saver.save(net, val_auc, epoch)
    writer.add_scalar("Test/AUC", val_auc, epoch)
    logger.info("Epoch: %d, Test AUC: %.4f, Valid AUC: %.4f, Best Valid AUC: %.4f", epoch, test_auc, val_auc, best_auc)

# final test with best validation model
with torch.no_grad():
    utils.load_pretrained(net, saver.last_checkpoint)
    net.cuda(args.device)
    net.eval()
aucs = []
for i in range(args.num_runs):
    auc = eval(net, test_loader)
    logger.info("[{}]/[{}] Run, Test AUC: {:.4f}".format(i + 1, args.num_runs, auc))
    aucs.append(auc)
logger.info("Test AUC: {:.4f} +- {:.4f}".format(np.mean(aucs), np.std(aucs)))
