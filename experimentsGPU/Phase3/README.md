# Phase 3 — Pretrained Classifier + Joint Fine-tuning + Learnable Mask

## Research Question

Does joint fine-tuning alter the frequency preference observed in Phase 1, or is the preference architecturally fixed?

In Phase 1, the classifier was frozen and only the mask was trained. The resulting mask reflects which frequency components the classifier, as a static function, relies on at inference time. In Phase 3, both the mask and the classifier are updated jointly, allowing co-adaptation. If the Phase 3 mask correlates highly with the Phase 1 mask, the frequency preference is an inductive bias of the architecture — present regardless of whether the classifier adapts. If the masks diverge, the preference is malleable through training.

---

## Experimental Setup

### Pipeline

```
Image → FFT → Learnable 2D Frequency Mask (224×224) → IFFT → Pretrained Classifier (Fine-tuned)
```

The classifier is initialised from pretrained ImageNet weights (not random). Both the mask and classifier parameters are updated during training, subject to the stage-wise schedule described in Phase 3.2.

### Mask Design

Identical to Phase 1: 50,176 parameters, normalisation enforcing mean = 1.0 per update step.

### Dataset

| Cache | Source split | Size | Train / Val |
|---|---|---|---|
| `imagenet_25k_cache` | validation | 25k | Used in Phase 3.0 |
| `imagenet_100k_cache` | train | 100k | 80k / 20k — used in Phase 3.1 and 3.2 |

Note: the 100k cache draws from the ImageNet **train** split rather than validation, providing a more diverse and larger training set than Phase 1 and Phase 2.

---

## Iterations

### Phase 3.0 — Direct Fine-tuning, 25k (Failed)

**Notebook:** `resnet18_phase3.ipynb`
**Results:** `results/resnet18_phase3/`

| Hyperparameter | Value |
|---|---|
| Dataset | 25k validation images |
| Mask LR | 0.005 |
| Classifier LR | 1e-4 |
| Early stopping patience | 10 |

**Outcome:**

| Metric | Value |
|---|---|
| Baseline accuracy | 65.90% |
| Best validation accuracy | 65.90% (never exceeded baseline) |
| Final training accuracy | 99.22% |
| Final validation accuracy | 59.28% |
| Train / val gap | ~40 percentage points |

Validation accuracy dropped every epoch from epoch 1. Early stopped at epoch 10.

**Root cause:** Classifier LR of 1e-4 is too high on a small dataset. The classifier overwrites pretrained features from the first weight update, discarding the representations it originally relied on before the mask has had any opportunity to settle. This is catastrophic forgetting occurring at the very first epoch.

---

### Phase 3.1 — Direct Fine-tuning, 100k (Failed)

**Notebook:** `resnet18_phase3.1.ipynb`
**Results:** `results/resnet18_phase3.1/`

**Change from Phase 3.0:** Dataset scaled from 25k to 100k (80k train / 20k val) to test whether data volume resolved the degradation.

| Hyperparameter | Value |
|---|---|
| Dataset | 100k train images (80k / 20k split) |
| Mask LR | 0.005 |
| Classifier LR | 1e-4 |
| Early stopping patience | 10 |

**Outcome:**

| Metric | Value |
|---|---|
| Baseline accuracy | 73.81% |
| Best validation accuracy | 73.81% (never exceeded baseline) |
| Val accuracy, epoch 1 | 67.14% |
| Val accuracy, epoch 10 | 60.70% |
| Final training accuracy | 93.95% |

Per-epoch validation history: 67.14 → 66.75 → 64.88 → 64.58 → 63.95 → 62.36 → 62.30 → 61.28 → 61.70 → 60.70

Validation starts 6.7 points below baseline at epoch 1 and continues to decline. Early stopped at epoch 10.

**Root cause:** More data did not resolve the issue. LR=1e-4 is still too high — the classifier modifies its weights significantly before the mask can compensate, and the joint system never recovers above baseline. The problem is the learning rate, not the data volume.

---

### Phase 3.2 — Two-Stage Warmup, 100k (Running)

**Notebook:** `resnet18_phase3.2.ipynb`
**Results:** `results/resnet18_phase3.2/`

**Key design change:** A two-stage training schedule decouples mask warmup from classifier fine-tuning. In Stage 1, the classifier is frozen (identical conditions to Phase 1), allowing the mask to reach a stable configuration before the classifier is allowed to adapt. In Stage 2, the classifier is unfrozen with a learning rate 10× lower than Phase 3.1.

#### Training Schedule

| Stage | Epochs | Classifier State | Mask LR | Classifier LR |
|---|---|---|---|---|
| Warmup | 1–10 | Frozen | 0.005 | — |
| Fine-tune | 11–60 | Trainable | 0.001 | 1e-5 |

Additional details:
- Early stopping patience: 15, reset at the stage 1 → stage 2 transition
- Mask normalisation (mean = 1.0) maintained throughout

**Rationale:**
- Stage 1 is an exact replication of Phase 1 conditions. If the mask converges to a similar pattern as in Phase 1, this provides evidence of architectural stability.
- Stage 2 uses LR=1e-5 (10× lower than all prior Phase 3 attempts) to ensure the classifier adapts incrementally rather than overwriting pretrained features.
- The mask LR reduction in Stage 2 (0.005 → 0.001) prevents the mask from destabilising a classifier that is now adapting slowly.

**Status:** Running on cluster for all 4 architectures (ResNet-18, ResNet-50, AlexNet, VGG-16).

---

## Next Steps

1. Confirm that Stage 1 (warmup) validation accuracy meets or exceeds the Phase 1 result for each architecture — this validates that the warmup mask is well-conditioned before Stage 2 begins.
2. Monitor Stage 2 to verify validation accuracy does not drop below baseline, confirming that LR=1e-5 avoids catastrophic forgetting.
3. On completion, compute Pearson correlation between the Phase 3.2 learned mask and the corresponding Phase 1 mask for each architecture.
   - High correlation (r > 0.7) → frequency preference is **inductive bias** (architecturally fixed)
   - Low correlation → preference is malleable through joint training
4. If Phase 3.2 is successful, extend the comparison across all four architectures to determine whether the ResNet vs. sequential architecture divergence observed in Phase 1 holds under fine-tuning conditions.

---

## Artifacts

| Directory | Phase | Status |
|---|---|---|
| `results/resnet18_phase3/` | Phase 3.0 | Failed |
| `results/resnet18_phase3.1/` | Phase 3.1 | Failed |
| `results/resnet18_phase3.2/` | Phase 3.2 | In progress |

Each result directory contains:

| File | Description |
|---|---|
| `learned_mask.pt` | Trained mask weights |
| `learned_mask.png` | Frequency mask visualisation |
| `training_history.pt` | Per-epoch loss and accuracy |
| `summary.txt` | Numerical results summary |
| `checkpoint.pt` | Resumable training state |
