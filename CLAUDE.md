# Frequency Analysis of Neural Networks

## Research Goal
Characterise the **inductive bias** of CNN and ViT architectures via learnable frequency masks, as a precursor to sim2real transfer analysis.
**Question:** Which frequency components do different neural networks rely on for classification — and does this preference arise from architecture (inductive bias) or training?

---

## Methodology

**Pipeline:** Image → FFT → Learnable 2D Frequency Mask (224×224) → IFFT → Classifier

| Phase | Classifier Init | Mask | Goal |
|-------|----------------|------|------|
| 1 | Pretrained (frozen) | Normalize (mean=1) | What frequencies do pretrained models use? |
| 2 | Random init (trainable) | Sigmoid (bounded 0–2) | What frequencies emerge when learning from scratch? |
| 3.2 | Pretrained + warmup unfreeze | Normalize (mean=1) | What preferences emerge when pretrained + mask co-adapt? |

---

## Project Structure

```
SIM2REAL/            ← local experiments (notebooks, Phase 1 results)
SIM2REAL_CLUSTER/    ← cluster-ready scripts (GPU training)
├── scripts/
│   ├── train_phase2.py        # Phase 2: from-scratch, sigmoid mask, DDP+AMP
│   ├── train_phase3_2.py      # Phase 3.2: warmup unfreeze, normalize mask, DDP+AMP
│   ├── distributed.py         # DDP/AMP utilities (one process per GPU)
│   ├── cache_imagenet_full.py # Cache full ImageNet (~1.28M train + 50k val)
│   └── cache_imagenet_100k.py # Legacy 100k subset cache
├── frequency/
│   ├── mask.py                # Learnable2DFrequencyMask (sigmoid + normalize modes)
│   ├── transforms.py
│   └── pipeline.py
├── experiments/results/       # Saved artifacts per arch/phase
├── run_phase2_full.sh         # Phase 2 on full ImageNet (DDP, sequential per arch)
└── run_experiments.sh         # Phase 3.2 jobs (one arch per GPU, 100k cache)
```

---

## Current Status

### Phase 1: Frozen Classifier (COMPLETE)
**Question:** "What frequencies do pre-trained ImageNet models use?"

| Architecture | Baseline | Final | Change |
|--------------|----------|-------|--------|
| ResNet-18    | 65.70%   | 67.24% | +1.54% |
| AlexNet      | 52.48%   | 47.48% | -5.00% |
| VGG-16       | 69.48%   | 65.77% | -3.70% |
| ResNet-50    | 74.18%   | 74.56% | +0.38% |

**Pattern:** ResNets improve/maintain; AlexNet and VGG degrade. Hypothesis: skip connections enable frequency-selective processing.

---

### Phase 2: Random Init + Sigmoid Mask
**Question:** "What frequency pattern emerges when learning from scratch?"

**Evolution of the recipe:**
- v1–v3 failed: mask collapse (unbounded weights), overfitting (25k data), no gradient clipping
- v4 (supervisor): sigmoid mask `2*sigmoid(w)` bounded (0,2) + 100k images + grad clip + cosine LR + 4-GPU DDP + AMP + linear LR scaling + persistent_workers/pin_memory; ViT uses AdamW + 5-epoch warmup
- v5 (current): same recipe, **full ImageNet** (~1.28M train + 50k val) + 90-epoch schedule + 5-epoch warmup default. Removes the "too little data" confound from the AlexNet/VGG failure.

**v4 (100k subset) results — kept for reference:**

| Architecture | Status | Val Top1 | Val Top5 | Epochs |
|--------------|--------|----------|----------|--------|
| ResNet-18    | DONE   | 30.98%   | 54.86%   | 120    |
| ResNet-18 (baseline variant) | DONE | 30.26% | 53.54% | 120 |
| AlexNet      | DONE   | 0.14%    | 0.50%    | 28 (early stop) |
| VGG-16       | DONE   | ~0.1%    | —        | early stop |
| ResNet-50    | DONE   | 33.16%   | 56.86%   | 120    |
| ViT-B/16     | RUNNING | —       | —        | —      |

**v5 (full ImageNet) — TODO** for ResNet-18, ResNet-50, AlexNet, VGG-16, ViT-B/16, and DenseNet-121 (added as a stronger skip-connection control).

**Key finding from v4:** ResNet architectures learn from scratch through the frequency pipeline (~30% top-1 on 100k); AlexNet and VGG-16 fail completely (flat mask, 0.1% accuracy). v5 will tell us whether the AlexNet/VGG failure is genuinely architectural or just data-limited.

---

### Phase 3.2: Pretrained + Warmup Unfreeze (IN PROGRESS on cluster)
**Question:** "What frequency preferences emerge when pretrained classifier and mask co-adapt?"

| Architecture | Status | Notes |
|--------------|--------|-------|
| ResNet-18    | RUNNING | Started on Neptun cluster |
| AlexNet      | TODO   | — |
| VGG-16       | TODO   | — |
| ResNet-50    | TODO   | — |
| ViT-B/16     | TODO   | — |

**Design:** Stage 1 (ep 1–10): classifier frozen, mask trains alone. Stage 2 (ep 11+): classifier unfrozen with LR=1e-5.
**Hypothesis:** If Phase 3.2 mask correlates with Phase 1 mask → frequency preference is inductive bias, not learned.

---

## Immediate TODOs

1. **Cache full ImageNet on the cluster:** `python3 scripts/cache_imagenet_full.py` (one-time, ~150 GB to `/mnt/local_learning/data/$USER/imagenet_full`)
2. **Re-run Phase 2 on full ImageNet** for all architectures (`bash run_phase2_full.sh`). Sequential, all GPUs per arch via DDP. ~1–2 days per CNN.
3. **Phase 4 (no retraining):** apply the existing Phase 1 / Phase 2 masks to a shifted-domain testbed (ImageNet-Sketch / ImageNet-R / Stylized-ImageNet) for sim2real evaluation; include FDA (Yang & Soatto 2020) as a hand-crafted baseline.
4. **Mask correlation analysis:** Phase 1 vs Phase 2 mask per architecture — answers whether frequency preference is inductive bias or pretraining artifact.

---

## Commands Reference (Cluster: Neptun)

```bash
# SSH
ssh bab61wot@neptun.cs.uni-kl.de
cd ~/SIM2REAL_CLUSTER

# Pull latest code
git pull

# One-time: cache full ImageNet (~150 GB to fast local NVMe)
huggingface-cli login
python3 scripts/cache_imagenet_full.py \
    --output /mnt/local_learning/data/$USER/imagenet_full

# Phase 2 (full ImageNet) — DDP, all GPUs per arch, sequential
nohup bash run_phase2_full.sh > logs/master_phase2_full.log 2>&1 &

# Single arch (testing) — uses 4-GPU DDP
torchrun --standalone --nproc_per_node=4 scripts/train_phase2.py \
    --arch resnet18 --data-dir /mnt/local_learning/data/$USER/imagenet_full

# Phase 3.2 (legacy 100k path)
nohup bash run_experiments.sh > logs/master.log 2>&1 &

# Monitor
tail -f logs/<arch>_phase2_full.log

# Check running jobs
ps aux | grep train_phase
```

**Data:** `/mnt/local_learning/data/$USER/imagenet_full` (full ImageNet, must be cached first). Legacy 100k path still works via `--data-dir /mnt/local_learning/data/$USER/imagenet_100k_cache`.
**Results:** `experiments/results/{arch}_phase2/` and `experiments/results/{arch}_phase3.2/`

---

## Technical Notes

- PyTorch 2.6+: Use `weights_only=False` in `torch.load()` for checkpoints
- Phase 2 mask: `activation='sigmoid'` — bounded (0, 2), explosion-proof
- Phase 3.2 mask: `normalize=True` — mean forced to 1.0 (fine for pretrained, no collapse risk)
- `train_phase2.py` auto-detects `DatasetDict` (full) vs `Dataset` (100k legacy)
- GPU utilization features (all in place): DDP one-process-per-GPU, AMP autocast+GradScaler, linear LR scaling × world_size, `persistent_workers=True` + `pin_memory=True`, ViT-B/16 uses AdamW + 5-epoch linear warmup
- Neptun cluster: 4× RTX 6000 Ada GPUs, shared home dir (no rsync needed for code)
- Always stage data on `/mnt/local_learning/data/$USER/` — home dir reads bottleneck training
- Script renamed: `train_phase3.2.py` → `train_phase3_2.py` (underscore)
