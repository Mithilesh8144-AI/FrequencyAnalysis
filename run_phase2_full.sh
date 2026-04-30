#!/bin/bash
# Phase 2 (full ImageNet) — DDP, all GPUs per architecture, sequential runs.
#
# Each architecture trains on ALL available GPUs via DDP (torchrun), then the
# next architecture starts. This is faster per-architecture than the
# old "1 GPU per arch in parallel" pattern — important for full ImageNet
# where a single-GPU run would take ~4-7 days per architecture.
#
# Before running:
#   1. Cache the full dataset to fast local storage (NOT in the project home):
#        python3 scripts/cache_imagenet_full.py \
#            --output /mnt/local_learning/data/$USER/imagenet_full
#   2. From SIM2REAL_CLUSTER:
#        nohup bash run_phase2_full.sh > logs/master_phase2_full.log 2>&1 &
#
# After all runs finish, sync results back and free up cluster local storage:
#   rm -rf /mnt/local_learning/data/$USER/imagenet_full

DATA_DIR="/mnt/local_learning/data/$USER/imagenet_full"
SCRIPT="scripts/train_phase2.py"
LOG_DIR="logs"
NPROC=${NPROC:-4}  # number of GPUs per run; override with `NPROC=N bash run_phase2_full.sh`

mkdir -p $LOG_DIR

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Dataset not found at $DATA_DIR"
    echo "Run scripts/cache_imagenet_full.py first."
    exit 1
fi

echo "Starting Phase 2 (full ImageNet) — DDP with $NPROC GPUs per arch, sequential."
echo "Data: $DATA_DIR"
echo ""

# Architectures to run, in order. Reorder / drop lines as needed.
ARCHS=(resnet18 resnet50 alexnet vgg16 densenet121 vit_b_16)

for arch in "${ARCHS[@]}"; do
    log="$LOG_DIR/${arch}_phase2_full.log"
    echo "[$(date '+%F %T')] Starting $arch  ->  $log"
    torchrun --standalone --nproc_per_node=$NPROC $SCRIPT \
        --arch $arch --data-dir $DATA_DIR > $log 2>&1
    echo "[$(date '+%F %T')] Finished $arch (exit $?)"
done

echo ""
echo "All Phase 2 (full ImageNet) jobs complete."
