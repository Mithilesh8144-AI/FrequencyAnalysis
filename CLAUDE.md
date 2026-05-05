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

**v5 (full ImageNet) status (as of 2026-05-05):**

| Architecture | Status | Notes |
|--------------|--------|-------|
| ResNet-18    | DRY-RUN VERIFIED | 1-epoch test passed: loss 6.93→6.08, top-1 1.9% mid-epoch, ~7 min/epoch on 4 GPUs |
| ResNet-50    | TODO | — |
| AlexNet      | TODO | Critical — tests data-limited vs architectural |
| VGG-16       | TODO | Same hypothesis as AlexNet |
| DenseNet-121 | TODO | New skip-connection control |
| ViT-B/16     | TODO | AdamW + 5-ep warmup wired |

User prefers single-arch runs (Option B in commands below) over the full sequential `run_phase2_full.sh` launcher.

**v4 result folders renamed to `*_phase2_v4_100k`** so v5 runs don't overwrite them. `resnet18_phase2_baseline` (no-FFT control) was deliberately not renamed.

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

1. ~~Cache full ImageNet on the cluster~~ **DONE.** Lives at `/mnt/local_learning/data/bab61wot/imagenet_full` — 156 GB, splits: train (1,281,167) + validation (50,000) + test (100,000, unused).
2. ~~Phase 2 v5 dry-run (resnet18, 1 epoch)~~ **VERIFIED 2026-05-05.** All 5 supervisor GPU features confirmed wired in.
3. **Phase 2 v5 real runs** — single arch at a time per user preference. Pattern: `nohup torchrun --standalone --nproc_per_node=4 scripts/train_phase2.py --arch <NAME> --data-dir /mnt/local_learning/data/$USER/imagenet_full > logs/<NAME>_phase2_full.log 2>&1 & disown`. Run order: resnet18 → resnet50 → alexnet → vgg16 → densenet121 → vit_b_16.
4. **Phase 4 (no retraining):** apply the existing Phase 1 / Phase 2 masks to a shifted-domain testbed (ImageNet-Sketch / ImageNet-R / Stylized-ImageNet) for sim2real evaluation; include FDA (Yang & Soatto 2020) as a hand-crafted baseline.
5. **Mask correlation analysis:** Phase 1 vs Phase 2 mask per architecture — answers whether frequency preference is inductive bias or pretraining artifact.

---

## Commands Reference (Cluster: Neptun)

**Important:** Neptun home dir is shared with the user's local PC (`~/VIT/SIM2REAL_CLUSTER/`). Code edits made locally are instantly visible on the cluster — **no `git pull` needed on the cluster**. Push to GitHub is for backup/sharing only.

```bash
# SSH (project is at ~/VIT/SIM2REAL_CLUSTER on cluster, NOT ~/SIM2REAL_CLUSTER)
ssh bab61wot@neptun.cs.uni-kl.de
cd ~/VIT/SIM2REAL_CLUSTER

# Optional but recommended: send Python tempdirs to local NVMe to avoid NFS .nfs* cleanup spam at process exit
export TMPDIR=/mnt/local_learning/data/$USER/tmp && mkdir -p $TMPDIR

# Phase 2 v5 — single arch (preferred), all 4 GPUs via DDP, 90 epochs default
ARCH=resnet18  # or resnet50, alexnet, vgg16, densenet121, vit_b_16
mkdir -p logs
nohup torchrun --standalone --nproc_per_node=4 scripts/train_phase2.py \
    --arch $ARCH --data-dir /mnt/local_learning/data/$USER/imagenet_full \
    > logs/${ARCH}_phase2_full.log 2>&1 &
disown

# Phase 2 v5 — full sequential launcher (all 6 archs back-to-back, ~5-7 days total)
nohup bash run_phase2_full.sh > logs/master_phase2_full.log 2>&1 &
disown

# Dry-run (single epoch, foreground — for pipeline verification)
torchrun --standalone --nproc_per_node=4 scripts/train_phase2.py --arch resnet18 --data-dir /mnt/local_learning/data/$USER/imagenet_full --epochs 1
# IMPORTANT: paste the dry-run on ONE line — backslash line continuations break with trailing whitespace

# Monitor
tail -f logs/${ARCH}_phase2_full.log              # follow current run
tail -50 logs/${ARCH}_phase2_full.log             # last 50 lines (after reconnecting)
grep "New best" logs/${ARCH}_phase2_full.log      # all val accuracy improvements
nvidia-smi                                        # GPU util, all 4 should be ~95%+
ps aux | grep train_phase2 | grep -v grep         # is training still alive?

# Stop training (graceful — checkpoint at last completed epoch is preserved)
pkill -f train_phase2.py
ps aux | grep train_phase2 | grep -v grep         # verify all 4 workers gone

# Resume after a stop — same command as launch above. Auto-loads checkpoint.pt
# from experiments/results/${ARCH}_phase2/ if present. Look for "Resuming from
# checkpoint at epoch N" in the log to confirm. Note: checkpoints are written
# at end-of-epoch, so killing mid-epoch loses that partial epoch.
```

**Disconnecting SSH while training:** after `disown`, training is fully detached from the shell. Press `Ctrl+C` to stop `tail`-ing, then `exit` to close SSH. Training keeps running.

**Data:** `/mnt/local_learning/data/bab61wot/imagenet_full` (already cached, 156 GB DatasetDict).
**Results:** `experiments/results/{arch}_phase2/` (v5 outputs) and `experiments/results/{arch}_phase2_v4_100k/` (v4 backups).
**Runbook:** `RUNBOOK.md` has the full step-by-step playbook.

---

## Technical Notes

- PyTorch 2.6+: Use `weights_only=False` in `torch.load()` for checkpoints
- Phase 2 mask: `activation='sigmoid'` — bounded (0, 2), explosion-proof
- Phase 3.2 mask: `normalize=True` — mean forced to 1.0 (fine for pretrained, no collapse risk)
- `train_phase2.py` auto-detects `DatasetDict` (full) vs `Dataset` (100k legacy)
- GPU utilization features (all in place, verified in v5 dry-run): DDP one-process-per-GPU, AMP autocast+GradScaler, linear LR scaling × world_size, `persistent_workers=True` + `pin_memory=True`, ViT-B/16 uses AdamW + 5-epoch linear warmup
- Neptun cluster: 4× RTX 6000 Ada GPUs, shared home dir (no rsync needed for code)
- Always stage data on `/mnt/local_learning/data/$USER/` — home dir reads bottleneck training
- Script renamed: `train_phase3.2.py` → `train_phase3_2.py` (underscore)
- `.gitignore` excludes all `*.pt` files — checkpoints/masks are not pushed to GitHub. Rerun training on cluster to regenerate.
- "Corrupt EXIF data" warnings during training are harmless (malformed metadata in some ImageNet images, PIL loads them fine)
- Project lives at `~/VIT/SIM2REAL_CLUSTER` on cluster (not `~/SIM2REAL_CLUSTER`)
- GitHub remote: `git@github.com:Mithilesh8144-AI/FrequencyAnalysis.git`
