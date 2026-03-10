# Phase 2 — Random Initialisation + Joint Training (Abandoned)

## Research Question

What frequency preferences emerge when a classifier and frequency mask are trained jointly from a random initialisation on ImageNet?

This is a controlled contrast to Phase 1: if a randomly initialised model converges on frequency preferences similar to those of a pretrained model (Phase 1), those preferences would be attributable to the training objective rather than to pretrained representations. If they diverge, pretrained representations are the causal factor.

---

## Experimental Setup

### Pipeline

```
Image → FFT → Learnable 2D Frequency Mask (224×224) → IFFT → Randomly Initialised Classifier (Trainable)
```

Both the mask and the classifier are updated during training. The classifier is ResNet-18 with randomly initialised weights (no pretrained loading).

### Dataset

| Property | Value |
|---|---|
| Source | ILSVRC/imagenet-1k (HuggingFace) |
| Split | Validation |
| Total size | 25,000 images |
| Train / val split | 20,000 / 5,000 (introduced from v2 onwards) |

---

## Iterations

Three versions were attempted, each addressing the failure mode of the previous.

### v1 — Baseline Design (Failed)

**Notebook:** `resnet18_phase2.ipynb`
**Results:** `results/resnet18_phase2/`

| Property | Value |
|---|---|
| Val split | None |
| Data augmentation | No |
| Mask collapse prevention | None |

**Outcome:** Training accuracy reached 99.91% within a few epochs. The mask collapsed to near-zero (mean ≈ 0.0). Correlation with the Phase 1 Phase 1 mask: −0.0003 (not meaningful).

**Root cause:** Without a validation split, the model memorised the 25k training images. No meaningful frequency selection occurred. The experimental design was invalid — there was no generalisation signal to drive mask learning.

---

### v2 — 80/20 Split + Regularisation (Failed)

**Notebook:** `resnet18_phase2_v2.ipynb`
**Results:** `results/resnet18_phase2_v2/`

**Changes from v1:**
- 80/20 train/val split introduced
- Data augmentation added (random crops, horizontal flips)
- Mask L2 regularisation added to discourage collapse
- Validation tracking and early stopping added

**Outcome:** The mask still collapsed despite regularisation.

**Root cause:** L2 regularisation penalises large mask values but does not prevent the mask from uniformly shrinking to zero. Regularisation alone is insufficient to maintain mask activity when the joint optimisation still finds it easier to route information through the classifier than through mask features.

---

### v3 — Mask Normalisation (Failed)

**Notebook:** `resnet18_phase2_v3.ipynb`
**Results:** `results/resnet18_phase2_v3/`

**Changes from v2:**
- Mask normalisation (mean forced to 1.0 after each update) replaces regularisation
- Regularisation removed

| Hyperparameter | Value |
|---|---|
| Mask LR | 0.01 |
| Classifier LR | 0.001 |
| Data augmentation | Yes |
| Mask normalisation | Enabled (mean = 1.0) |

**Outcome:** Mask normalisation successfully prevented collapse. However, the model massively overfit.

| Metric | Value |
|---|---|
| Epochs trained | 54 (early stopped, patience = 15) |
| Best validation accuracy | 9.32% |
| Final training accuracy | 93.55% |
| Final validation accuracy | 8.82% |
| Train / val gap | ~84 percentage points |

**Root cause:** ResNet-18 has 11.7 million parameters. Training from scratch requires approximately 500k+ samples to generalise; 20k images are insufficient by one to two orders of magnitude. The model memorised training examples without learning transferable features, and the mask adapted to the memorised representations rather than to generalisable frequency structure.

---

## Lessons Learned

1. **Data scale:** 20k images is wholly inadequate for training a ResNet-18 from scratch. Meaningful from-scratch training requires at minimum ~500k images.
2. **Mask normalisation:** Forcing the mask mean to 1.0 is more effective than L2 regularisation for preventing collapse. This technique was carried forward to Phase 3.
3. **Pretrained weights are necessary:** Without pretrained representations, the classifier cannot provide useful gradients to the mask on a small dataset. Phase 2 provides no evidence about frequency preferences because the model never learns to classify.

---

## Why Phase 2 Was Abandoned

The fundamental constraint is data volume, not experimental design. Resolving the overfitting in v3 would require access to the full ImageNet training set (1.2M images) and substantially longer training, which is outside the scope of this project. More importantly, training from scratch conflates the inductive bias question (Phase 1) with a data-sufficiency question.

Phase 2 was abandoned and the project redirected to **Phase 3**: use pretrained weights as the classifier starting point, then jointly fine-tune the classifier and mask, allowing co-adaptation starting from a meaningful feature space rather than random noise.
