# Frequency Analysis — Cluster Training

Training code for **Phase 3.2** of the frequency analysis project.

## Research Goal

Understand the **inductive bias** of different CNN architectures by learning which frequency components each model relies on for classification. This is a step towards sim2real transfer analysis.

**Pipeline:** Image → FFT → Learnable Frequency Mask → IFFT → Classifier

## What is Phase 3.2?

Phase 3.2 answers: *"Does fine-tuning change the frequency preference, or is it truly baked into the architecture?"*

It uses a **two-stage warmup strategy** to prevent catastrophic forgetting:

| Stage | Epochs | Classifier | Mask LR | Classifier LR |
|-------|--------|------------|---------|---------------|
| Warmup | 1–10 | Frozen | 0.005 | — |
| Fine-tune | 11–60 | Trainable | 0.001 | 1e-5 |

- **Stage 1**: Classifier frozen. Mask learns frequency preferences without disturbing pretrained weights (same as Phase 1).
- **Stage 2**: Classifier unfrozen with very low LR so it adapts gently alongside the mask.

Comparing the Phase 3.2 mask with the Phase 1 mask reveals whether the frequency preference is architectural (inductive bias) or changes with joint training.

## Project Structure

```
FrequencyAnalysis/
├── scripts/
│   └── train_phase3.2.py     # Main training script (supports all 4 architectures)
├── frequency/
│   ├── transforms.py         # FFT / IFFT wrappers
│   ├── mask.py               # Learnable2DFrequencyMask
│   └── pipeline.py           # Phase 1 frozen pipeline
├── data/
│   └── dataset.py            # HuggingFace ImageNet dataset wrapper
├── run_experiments.sh        # Launches all 4 architectures in parallel (one per GPU)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Requires `imagenet_100k_cache` — 100k ImageNet train images pre-cached with HuggingFace datasets.

## Running

**Single architecture:**
```bash
python3 scripts/train_phase3.2.py --arch resnet18 --data-dir /path/to/imagenet_100k_cache
```

**All 4 architectures in parallel (one per GPU):**
```bash
nohup bash run_experiments.sh &
```

Supported architectures: `resnet18`, `resnet50`, `alexnet`, `vgg16`

## Outputs

Results saved to `experiments/results/{arch}_phase3.2/`:

| File | Description |
|------|-------------|
| `checkpoint.pt` | Full training state — auto-resumes if interrupted |
| `best_model.pt` | Best validation accuracy checkpoint |
| `learned_mask.pt` | Final learned frequency mask weights |
| `learned_mask_viz.npy` | Mask as numpy array for analysis |
| `training_history.pt` | Loss/accuracy history per epoch |
| `training_history.png` | Training curves plot |
| `learned_mask.png` | Frequency mask visualization |
| `summary.txt` | Results summary |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Warmup epochs | 10 |
| Total epochs (max) | 60 |
| Early stopping patience | 15 |
| Mask LR (warmup) | 0.005 |
| Mask LR (finetune) | 0.001 |
| Classifier LR (finetune) | 1e-5 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Train/Val split | 80/20 |
| Dataset | ImageNet 100k (train split) |
