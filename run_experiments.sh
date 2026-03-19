#!/bin/bash
# Run all Phase 3.2 experiments in parallel, one per GPU.
# Usage: bash run_experiments.sh <your-neptun-username>
#
# Before running:
#   1. Copy dataset: rsync -avz --progress data/imagenet_100k_cache/ \
#        <user>@neptun.cs.uni-kl.de:/mnt/local_learning/data/<user>/imagenet_100k_cache/
#   2. Copy this folder to Neptun
#   3. Run: conda activate <your-env> && bash run_experiments.sh

DATA_DIR="/mnt/local_learning/data/$USER/imagenet_100k_cache"
SCRIPT="scripts/train_phase3_2.py"
LOG_DIR="logs"

mkdir -p $LOG_DIR

echo "Starting Phase 3.2 experiments on 5 GPUs..."
echo "Data: $DATA_DIR"
echo ""

# Run one architecture per GPU
CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT --arch resnet18 --data-dir $DATA_DIR > $LOG_DIR/resnet18.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 $SCRIPT --arch alexnet  --data-dir $DATA_DIR > $LOG_DIR/alexnet.log   2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT --arch vgg16    --data-dir $DATA_DIR > $LOG_DIR/vgg16.log     2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 $SCRIPT --arch resnet50 --data-dir $DATA_DIR > $LOG_DIR/resnet50.log  2>&1 &
CUDA_VISIBLE_DEVICES=4 python3 $SCRIPT --arch vit_b_16 --data-dir $DATA_DIR > $LOG_DIR/vit_b_16.log  2>&1 &

echo "All 5 jobs launched. Monitor with:"
echo "  tail -f logs/resnet18.log"
echo "  tail -f logs/alexnet.log"
echo "  tail -f logs/vgg16.log"
echo "  tail -f logs/resnet50.log"
echo "  tail -f logs/vit_b_16.log"

# Wait for all to finish
wait
echo ""
echo "All 5 experiments complete!"
echo ""
echo "IMPORTANT: Copy results back and delete local data:"
echo "  rsync -avz experiments/results/ <your-pc>:~/VIT/SIM2REAL/experiments/results/"
echo "  rm -rf $DATA_DIR"
