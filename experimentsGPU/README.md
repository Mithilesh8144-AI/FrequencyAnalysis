# Frequency Analysis of Neural Networks — Experiment Log

## Overview

This project investigates the **frequency domain preferences** of convolutional neural networks as a means of characterising their inductive bias. The central question is: *which frequency components do different architectures rely on for ImageNet classification?*

Understanding these preferences is a precursor to sim2real transfer analysis, where synthetic and real images differ systematically in their frequency content.

**Pipeline:**
```
Image → FFT → Learnable 2D Frequency Mask (224×224) → IFFT → Classifier
```

The mask has 50,176 learnable parameters. The classifier provides gradients. By examining what the mask learns, we can quantify what frequency information each architecture requires.

---

## Experiment Phases

### Phase 1 — Frozen Classifier (Complete)

**Research question:** What frequency components do pre-trained ImageNet classifiers rely on?

**Setup:**
- Pretrained classifier, frozen throughout
- Only the frequency mask is trained (50,176 params)
- Dataset: 25k ImageNet validation images
- Mask normalisation (mean=1.0) to prevent weight collapse

| Architecture | Baseline | After Mask | Change | Notebook |
|---|---|---|---|---|
| ResNet-18 | 65.70% | 67.24% | **+1.54%** | `Phase1/resnet18_experiment3_fixed.ipynb` |
| ResNet-50 | 74.18% | 74.56% | **+0.38%** | `Phase1/resnet50_experiment3.ipynb` |
| AlexNet | 52.48% | 47.48% | **−5.00%** | `Phase1/alexnet_experiment3.ipynb` |
| VGG-16 | 69.48% | 65.77% | **−3.70%** | `Phase1/vgg16_experiment3.ipynb` |

**Finding:** ResNet architectures (skip connections) maintain or improve accuracy under frequency filtering. AlexNet and VGG-16 (sequential architectures) degrade. This suggests skip connections enable frequency-selective processing as an inductive bias.

**Artifacts:** `results/{resnet18, resnet50, alexnet, vgg16}/`

---

### Phase 2 — Random Initialisation + Joint Training (Abandoned)

**Research question:** What frequency preferences emerge when a model learns from scratch?

**Setup:** Randomly initialised ResNet-18 + trainable mask, 25k images

Three iterations were attempted, each addressing the failure mode of the previous:

| Version | Key Change | Outcome | Root Cause |
|---|---|---|---|
| v1 | Baseline design | 99.91% train, mask collapsed | No val split — pure memorisation |
| v2 | 80/20 split, augmentation, regularisation | Mask still collapsed | Regularisation insufficient |
| v3 | Mask normalisation (mean=1.0) | 93.55% train / 9.32% val | 20k images insufficient for 11.7M params |

**Conclusion:** Training from scratch requires orders of magnitude more data (~500k+). Pretrained weights are necessary. Phase 2 was abandoned.

**Artifacts:** `results/resnet18_phase2*/`

---

### Phase 3 — Pretrained Classifier + Joint Fine-tuning (In Progress)

**Research question:** Does joint fine-tuning alter the frequency preference observed in Phase 1, or is it architecturally fixed?

If Phase 3 masks correlate highly with Phase 1 masks → frequency preference is **inductive bias** (architectural).
If they diverge → preference is malleable through training.

#### Phase 3.0 — Direct Fine-tuning, 25k (Failed)
- Classifier LR: 1e-4, data: 25k
- Val dropped every epoch from epoch 1 (99% train / 59% val)
- Root cause: LR too high — catastrophic forgetting from epoch 1

#### Phase 3.1 — Direct Fine-tuning, 100k (Failed)
- Classifier LR: 1e-4, data: 100k (80k train / 20k val)
- Val history: 67.14 → 60.70 over 10 epochs (baseline: 73.81%)
- Root cause: LR=1e-4 still too high — same catastrophic forgetting

#### Phase 3.2 — Two-Stage Warmup, 100k (Running)
- **Notebook:** `Phase3/resnet18_phase3.2.ipynb`
- **Key fix:** Classifier frozen for first 10 epochs (identical to Phase 1), then unfrozen with LR=1e-5

| Stage | Epochs | Classifier | Mask LR | Classifier LR |
|---|---|---|---|---|
| Warmup | 1–10 | Frozen | 0.005 | — |
| Fine-tune | 11–60 | Trainable | 0.001 | 1e-5 |

- Warmup lets the mask settle before the classifier is allowed to adapt
- LR=1e-5 (10× lower than Phase 3.1) prevents overwriting of pretrained features
- Early stopping patience: 15, reset at stage transition
- Currently running on cluster for all 4 architectures

**Artifacts:** `results/resnet18_phase3.2/`

---

## Results Structure

```
results/
├── resnet18/               ← Phase 1 results
├── alexnet/                ← Phase 1 results
├── vgg16/                  ← Phase 1 results
├── resnet50/               ← Phase 1 results
├── resnet18_phase2/        ← Phase 2 v1 (failed)
├── resnet18_phase2_v2/     ← Phase 2 v2 (failed)
├── resnet18_phase2_v3/     ← Phase 2 v3 (failed)
├── resnet18_phase3/        ← Phase 3.0 (failed)
├── resnet18_phase3.1/      ← Phase 3.1 (failed)
└── resnet18_phase3.2/      ← Phase 3.2 (in progress)
```

Each result directory contains:

| File | Description |
|---|---|
| `learned_mask.pt` | Trained mask weights |
| `learned_mask.png` | Frequency mask visualisation |
| `training_history.pt` | Loss and accuracy per epoch |
| `summary.txt` | Numerical results summary |
| `checkpoint.pt` | Resumable training state (Phase 3 only) |

---

## Dataset

| Cache | Split | Size | Used in |
|---|---|---|---|
| `imagenet_25k_cache` | validation | 25k | Phase 1, Phase 2 |
| `imagenet_100k_cache` | train | 100k | Phase 3 |

All data sourced from `ILSVRC/imagenet-1k` via HuggingFace datasets.
