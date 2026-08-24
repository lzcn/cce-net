#!/usr/bin/env bash
# Train and evaluate the hybrid CCE-Net (InfoNCE) on PolyVore (disjoint split).
set -e

FILE=train_cce_infonce.py
DATA_DIR="../outfits-hf"
DATA_SET="polyvore-outfits/disjoint"
BETA=0.5
LOG_DIR="runs/polyvore-outfits-d-infonce-fuse-beta-$BETA"
NUMNEG=4
TAU=0.10
FLAGS="--norm --feat-norm --neg-type-aware --trans-layer --fuse-score --fuse-beta $BETA"
OPTS="--tau $TAU --base-model GATConv --batch-size 128 --lr 0.1 --num-layers 2 --neg-ratio 1 --num-neg $NUMNEG"

# training
python $FILE --data-dir $DATA_DIR --data-set $DATA_SET --log-dir $LOG_DIR $FLAGS $OPTS --neg-mode RandomMix "$@"

# testing
python $FILE --data-dir $DATA_DIR --data-set $DATA_SET --log-dir $LOG_DIR $FLAGS $OPTS \
    --neg-mode RandomMix --test --load-trained $LOG_DIR/checkpoints/net_best.pt --log-name replace-all
