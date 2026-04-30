# Phase 2 (Full ImageNet) Runbook

Step-by-step playbook for re-running Phase 2 on full ImageNet. Reusable —
follow it top-to-bottom for fresh runs, or jump to the section you need.

> **Note:** The Neptun home directory is shared with your local PC, so any
> code in `~/SIM2REAL_CLUSTER` (or `~/VIT/SIM2REAL_CLUSTER`) is already in
> sync. You don't need git push/pull to deploy code changes.

---

## 1. SSH into Neptun

```bash
ssh bab61wot@neptun.cs.uni-kl.de
```

## 2. Navigate to the project folder

```bash
cd ~/SIM2REAL_CLUSTER 2>/dev/null || cd ~/VIT/SIM2REAL_CLUSTER
pwd
ls run_phase2_full.sh scripts/cache_imagenet_full.py
```

If both files are listed you're in the right place. If not:

```bash
find ~ -name run_phase2_full.sh 2>/dev/null
```

## 3. Install / refresh dependencies (one-time, skip if already done)

```bash
pip3 install -r requirements.txt
pip3 install --user huggingface_hub
```

## 4. HuggingFace auth (one-time)

```bash
huggingface-cli login
```

- Token: https://huggingface.co/settings/tokens (any read token)
- Accept dataset license: https://huggingface.co/datasets/ILSVRC/imagenet-1k

## 5. Create the local staging folder & verify free space

```bash
mkdir -p /mnt/local_learning/data/$USER
df -h /mnt/local_learning   # need ≥ 200 GB free
```

## 6. Download + cache full ImageNet (a few hours, run in tmux)

```bash
tmux new -s imagenet
python3 scripts/cache_imagenet_full.py \
    --output /mnt/local_learning/data/$USER/imagenet_full \
    --hf-cache-dir /mnt/local_learning/data/$USER/hf_cache
```

- Detach: `Ctrl+B` then `D`
- Reattach later: `tmux attach -t imagenet`

## 7. Verify the cache

```bash
ls /mnt/local_learning/data/$USER/imagenet_full/
du -sh /mnt/local_learning/data/$USER/imagenet_full/
```

You should see `train/`, `validation/`, and `dataset_dict.json`. Total size
should be roughly 150 GB.

## 8. (Recommended) Back up the existing v4 (100k) Phase 2 results

The new runs reuse the same `experiments/results/{arch}_phase2/` folders.
Rename the old ones first so they aren't overwritten.

```bash
cd ~/SIM2REAL_CLUSTER
for arch in resnet18 resnet50 alexnet vgg16 resnet18_baseline; do
    src="experiments/results/${arch}_phase2"
    [ -d "$src" ] && mv "$src" "${src%_phase2}_phase2_v4_100k"
done
ls experiments/results/
```

## 9. Quick dry-run (10–15 min) before the long launch

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_phase2.py \
    --arch resnet18 \
    --data-dir /mnt/local_learning/data/$USER/imagenet_full \
    --epochs 1
```

If it errors out, share the log and fix before continuing. If it works,
`Ctrl+C` and remove the partial result:

```bash
rm -rf experiments/results/resnet18_phase2
```

## 10. Launch the full Phase 2 run (sequential, all GPUs per arch)

```bash
mkdir -p logs
nohup bash run_phase2_full.sh > logs/master_phase2_full.log 2>&1 &
disown
```

By default the script runs `resnet18 → resnet50 → alexnet → vgg16 →
densenet121 → vit_b_16` sequentially, each using all 4 GPUs.

To run on fewer GPUs:

```bash
NPROC=2 nohup bash run_phase2_full.sh > logs/master_phase2_full.log 2>&1 &
```

To skip / reorder architectures, edit the `ARCHS=(...)` line in
`run_phase2_full.sh`.

## 11. Monitor

```bash
# Master log: which arch is currently running, when each starts/finishes
tail -f logs/master_phase2_full.log

# Per-arch training log
tail -f logs/resnet18_phase2_full.log

# GPU utilization
nvidia-smi

# Running processes
ps aux | grep train_phase2
```

`Ctrl+C` only stops tailing — training keeps running. You can disconnect SSH
at any point.

## 12. Finished — clean up cluster local storage

Results live under `experiments/results/{arch}_phase2/`, which is on the
shared home dir, so they're already on your local PC.

```bash
rm -rf /mnt/local_learning/data/$USER/imagenet_full
rm -rf /mnt/local_learning/data/$USER/hf_cache
```

---

## Common operations

### Resume a single architecture from its last checkpoint

The training script auto-resumes from `experiments/results/{arch}_phase2/checkpoint.pt`
when present. Just relaunch the same command:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_phase2.py \
    --arch resnet18 \
    --data-dir /mnt/local_learning/data/$USER/imagenet_full
```

### Kill all running training jobs

```bash
pkill -f train_phase2.py
```

### Run a single architecture only (one-off, not via the launcher)

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_phase2.py \
    --arch densenet121 \
    --data-dir /mnt/local_learning/data/$USER/imagenet_full \
    > logs/densenet121_phase2_full.log 2>&1 &
```

### Run a baseline (no FFT/mask) for sanity comparison

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_phase2.py \
    --arch resnet18 --skip-fft \
    --data-dir /mnt/local_learning/data/$USER/imagenet_full
```

Output goes to `experiments/results/resnet18_phase2_baseline/`.

### Override the epoch budget

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_phase2.py \
    --arch resnet18 --epochs 30 \
    --data-dir /mnt/local_learning/data/$USER/imagenet_full
```

---

## What's wired in (no setup needed, for reference)

| Feature | Where |
|---|---|
| DDP — one process per GPU | `scripts/distributed.py` |
| AMP — `torch.amp.autocast` + `GradScaler` | `scripts/train_phase2.py` |
| Linear LR scaling × `world_size` (mask + classifier) | `scripts/train_phase2.py` |
| `persistent_workers=True`, `pin_memory=True` | `scripts/train_phase2.py` |
| ViT-B/16 → AdamW + 5-epoch linear warmup | `ARCH_HPARAMS` in `train_phase2.py` |
| Auto-detect full ImageNet (`DatasetDict`) vs legacy 100k (single `Dataset`) | `train_phase2.py` |
