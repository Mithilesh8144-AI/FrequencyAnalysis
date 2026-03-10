# Experiment Log

Detailed record of all experiments run for the frequency analysis project.

---

## Phase 1: Frozen Classifier + Learnable Mask (COMPLETE)

**Setup:** Pretrained ImageNet classifier (frozen) + learnable frequency mask
**Data:** 25k ImageNet validation images
**Question:** What frequencies do pre-trained models rely on?

### ResNet-18
- **Notebook:** `Phase1/resnet18_experiment3_fixed.ipynb`
- **Results:** `results/resnet18/`
- Baseline: 65.70% → After mask: 67.24% (+1.54%)
- Mask boosted low-mid frequencies, slightly suppressed high frequencies

### AlexNet
- **Notebook:** `Phase1/alexnet_experiment3.ipynb`
- **Results:** `results/alexnet/`
- Baseline: 52.48% → After mask: 47.48% (-5.00%)
- Accuracy dropped — AlexNet relies on broad frequency spectrum

### VGG-16
- **Notebook:** `Phase1/vgg16_experiment3.ipynb`
- **Results:** `results/vgg16/`
- Baseline: 69.48% → After mask: 65.77% (-3.70%)
- Similar to AlexNet — frequency filtering hurts sequential architectures

### ResNet-50
- **Notebook:** `Phase1/resnet50_experiment3.ipynb`
- **Results:** `results/resnet50/`
- Baseline: 74.18% → After mask: 74.56% (+0.38%)
- Slight improvement — consistent with ResNet-18 pattern

### Phase 1 Key Finding
- ResNets (skip connections) improve or maintain accuracy with frequency filtering
- AlexNet/VGG (sequential) degrade with frequency filtering
- Hypothesis: Skip connections enable frequency-selective processing (inductive bias)

---

## Phase 2: Random Init Classifier + Learnable Mask (FAILED)

**Setup:** Randomly initialized classifier (trainable) + learnable frequency mask
**Data:** 25k ImageNet validation images (20k train / 5k val)
**Question:** What frequencies does a model learn to prefer from scratch?

### Phase 2 v1 — ResNet-18 (FAILED)
- **Notebook:** `Phase2/resnet18_phase2.ipynb`
- **Results:** `results/resnet18_phase2/`
- Train accuracy: 99.91% — but this was pure memorization
- No train/val split, no augmentation, no validation tracking
- Mask collapsed to near-zero (mean ~0.0)
- Correlation with Phase 1 mask: -0.0003 (meaningless)
- **Conclusion:** Invalid results, experimental design flawed

### Phase 2 v2 — ResNet-18 (FAILED)
- **Notebook:** `Phase2/resnet18_phase2_v2.ipynb`
- **Results:** `results/resnet18_phase2_v2/`
- Fixes: 80/20 split, augmentation, validation tracking, early stopping
- Added mask regularization to prevent collapse
- Mask still collapsed despite regularization
- **Conclusion:** Regularization alone wasn't enough

### Phase 2 v3 — ResNet-18 (FAILED)
- **Notebook:** `Phase2/resnet18_phase2_v3.ipynb`
- **Results:** `results/resnet18_phase2_v3/`
- Key change: Mask normalization (mean forced to 1.0) instead of regularization
- Normalization prevented mask collapse
- Train: 93.55% | Val: 9.32% — massive overfitting
- Trained 54 epochs, early stopped at patience=15
- **Conclusion:** Training from scratch on 20k images with 11.7M params doesn't work

### Phase 2 Lessons Learned
1. 20k images is far too few for training ResNet-18 from scratch (needs ~500k+)
2. Mask normalization works better than regularization for preventing collapse
3. Need pretrained weights as starting point (→ led to Phase 3)

---

## Phase 3: Pretrained Classifier + Fine-tuning + Learnable Mask (IN PROGRESS)

**Setup:** Pretrained ImageNet classifier (fine-tuned) + learnable normalized mask
**Question:** What frequencies emerge when mask and pretrained classifier co-adapt?

### Phase 3 — ResNet-18 with 25k data (FAILED)
- **Notebook:** `Phase3/resnet18_phase3.ipynb`
- **Results:** `results/resnet18_phase3/`
- Pretrained weights + fine-tuning + normalized mask
- Baseline: 65.90% | Best val: 65.90% (never improved past baseline)
- Train: 99.22% | Val: 59.28% (last epoch) | Gap: 40%
- Val accuracy dropped every single epoch from epoch 1
- Early stopped at epoch 10 (patience=10)
- LRs: Mask 0.005, Classifier 0.0001

### Phase 3 — ResNet-18 with 25k data (FAILED)
- **Notebook:** `Phase3/resnet18_phase3.ipynb`
- **Results:** `results/resnet18_phase3/`
- Baseline: 65.90% | Best val: 65.90% (never improved past baseline)
- Train: 99.22% | Val: 59.28% (last epoch) | Gap: 40%
- Val accuracy dropped every single epoch from epoch 1
- Early stopped at epoch 10 (patience=10)
- LRs: Mask 0.005, Classifier 0.0001
- **Root cause:** 25k images too few AND classifier LR too high

### Phase 3.1 — ResNet-18 with 100k data (FAILED)
- **Notebook:** `Phase3/resnet18_phase3.1.ipynb`
- **Results:** `results/resnet18_phase3.1/`
- Upgraded to 100k ImageNet train images (80k train / 20k val)
- Baseline: 73.81% | Best val: 73.81% (never improved past baseline)
- Val acc history: [67.14, 66.75, 64.88, 64.58, 63.95, 62.36, 62.30, 61.28, 61.70, 60.70]
- Train acc history: [69.06, 75.17, 79.65, 83.10, 85.76, 88.31, 90.30, 91.56, 92.97, 93.95]
- Early stopped at epoch 10 (patience=10)
- **Root cause:** Val dropped from epoch 1 — classifier LR (0.0001) too high, overwrites pretrained features immediately

### Phase 3.2 — ResNet-18 with 100k data + Warmup (PENDING)
- **Notebook:** `Phase3/resnet18_phase3.2.ipynb`
- **Results:** `results/resnet18_phase3.2/`
- **Key change:** Two-stage training:
  - Stage 1 (epochs 1–10): Classifier FROZEN, mask trains alone (like Phase 1)
  - Stage 2 (epoch 11+): Classifier unfrozen with LR=1e-5 (10x lower than Phase 3.1)
- Mask LR: 0.005 (warmup) → 0.001 (finetune)
- Early stopping patience: 15 (reset at stage transition)
- **Status:** Ready to run

---

## Data Caches

| Cache | Split | Size | Used By |
|-------|-------|------|---------|
| `data/imagenet_10k_cache/` | validation | 10k | Unused |
| `data/imagenet_25k_cache/` | validation | 25k | Phase 1, Phase 2 |
| `data/imagenet_100k_cache/` | train | 100k | Phase 3 (new) |

---

## Experiment Timeline

1. Phase 1 (all 4 architectures) → Complete, clean results
2. Phase 2 v1 → Failed (memorization, no validation)
3. Phase 2 v2 → Failed (mask collapse despite regularization)
4. Phase 2 v3 → Failed (overfitting, 93% train / 9% val)
5. Phase 3 (25k, LR=1e-4) → Failed (val dropped every epoch from epoch 1)
6. Phase 3.1 (100k, LR=1e-4) → Failed (val starts below baseline, drops to 61%; LR too high)
7. Phase 3.2 (100k, warmup 10ep + LR=1e-5) → **Pending** (next experiment to run)
