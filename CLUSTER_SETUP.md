# Cluster Setup & Run Guide

## 1. SSH into Neptun

```bash
ssh bab61wot@neptun.cs.uni-kl.de
cd ~/SIM2REAL_CLUSTER
git pull
pip3 install -r requirements.txt
```

---

## 2. Stage the dataset on fast local storage

**Important:** the project home directory (`~/`) is shared/slow. Reading the
dataset from there during training will bottleneck GPU throughput. Always
cache to `/mnt/local_learning/data/$USER/`, which is local NVMe on the GPU
node.

### Full ImageNet (default for Phase 2 from-scratch runs)

```bash
huggingface-cli login   # one-time, needed for the gated dataset
python3 scripts/cache_imagenet_full.py \
    --output /mnt/local_learning/data/$USER/imagenet_full
```

This downloads both `train` (~1.28M) and `validation` (50k) splits and saves
them as a HuggingFace `DatasetDict`. Disk usage is ~150 GB. It takes a few
hours the first time, then is reused for every run.

### Legacy 100k cache (only if reproducing the old v4 results)

```bash
mkdir -p /mnt/local_learning/data/$USER
cp -r ~/VIT/SIM2REAL/data/imagenet_100k_cache/ /mnt/local_learning/data/$USER/
```

`train_phase2.py` auto-detects which format is at `--data-dir`:
- `DatasetDict` with `train`/`validation` → uses the native ImageNet split
- single `Dataset` → falls back to an 80/20 random split (legacy 100k path)

---

## 3. Launch Phase 2 (full ImageNet)

All architectures, all GPUs per run via DDP, sequential:

```bash
nohup bash run_phase2_full.sh > logs/master_phase2_full.log 2>&1 &
```

Override the GPU count if needed:

```bash
NPROC=2 bash run_phase2_full.sh
```

### Single architecture (testing or one-off)

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_phase2.py \
    --arch resnet18 \
    --data-dir /mnt/local_learning/data/$USER/imagenet_full
```

`--epochs N` overrides the default schedule (90 epochs + 5-epoch linear warmup).
`--skip-fft` runs the same architecture without the frequency pipeline as a
baseline.

---

## 4. GPU utilization features (already wired in)

Both `train_phase2.py` and `train_phase3_2.py` use the supervisor's
multi-GPU stack:

1. **DDP** (`scripts/distributed.py`) — one process per GPU, batch size scales
   with world size (64 → 256 effective on 4 GPUs).
2. **AMP** (`torch.amp.autocast` + `GradScaler`) — fp16 on the fly, less
   memory and faster matmuls. Disable with `--no-amp` if needed.
3. **Linear LR scaling** — mask LR and classifier LR are multiplied by
   `world_size` automatically.
4. **DataLoader: `persistent_workers=True`, `pin_memory=True`** — keeps
   worker processes alive between epochs and pre-pins tensors for faster
   host-to-GPU copies.
5. **ViT-B/16 specifics** — uses **AdamW** plus a 5-epoch linear LR warmup
   (`SequentialLR(LinearLR -> CosineAnnealingLR)`); transformers don't
   tolerate large LR spikes at step 0.

---

## 5. Launch Phase 3.2 (legacy 100k path)

```bash
nohup bash run_experiments.sh > logs/master_phase3.log 2>&1 &
```

This still runs against the 100k cache by default — see the script header.

---

## 6. Monitor

```bash
tail -f logs/resnet18_phase2_full.log
```

Press `Ctrl+C` to stop tailing. Training continues in the background.

---

## 7. Clean up after training

```bash
# Sync results back to your local machine first
rsync -avz experiments/results/ <your-pc>:~/VIT/SIM2REAL_CLUSTER/experiments/results/

# Then free up the cluster's local storage
rm -rf /mnt/local_learning/data/$USER/imagenet_full
```

Results live under `experiments/results/{arch}_phase2/` on your home
directory, which is persistent across sessions.
