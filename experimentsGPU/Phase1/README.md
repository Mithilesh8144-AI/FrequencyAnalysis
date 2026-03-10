# Phase 1 — Frozen Classifier + Learnable Frequency Mask

## Research Question

Which frequency components do pre-trained ImageNet classifiers rely on for classification?

---

## Experimental Setup

### Pipeline

```
Image → FFT → Learnable 2D Frequency Mask (224×224) → IFFT → Frozen Classifier
```

The mask is applied element-wise in the frequency domain. Only the mask parameters are updated during training; the classifier is frozen throughout. By inspecting what the mask learns to amplify or suppress, we can characterise each architecture's frequency preference.

### Mask Design

| Property | Value |
|---|---|
| Mask shape | 224 × 224 (2D, matching FFT output) |
| Trainable parameters | 50,176 |
| Normalisation | Mean forced to 1.0 each step to prevent weight collapse |
| Initialisation | Uniform 1.0 (identity — no initial filtering) |

Mask normalisation is critical: without it, the mask trivially collapses to near-zero under gradient descent, passing no information to the frozen classifier.

### Dataset

| Property | Value |
|---|---|
| Source | ILSVRC/imagenet-1k (HuggingFace) |
| Split | Validation |
| Size | 25,000 images |
| Classes | 1,000 (ImageNet standard) |

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate (mask) | 0.01 |
| Optimiser | Adam |
| Epochs | 20 |
| Batch size | 64 |
| Classifier | Frozen (pretrained ImageNet weights) |

---

## Results

| Architecture | Baseline Accuracy | Post-Mask Accuracy | Change | Notebook |
|---|---|---|---|---|
| ResNet-18 | 65.70% | 67.24% | **+1.54%** | `resnet18_experiment3_fixed.ipynb` |
| ResNet-50 | 74.18% | 74.56% | **+0.38%** | `resnet50_experiment3.ipynb` |
| AlexNet | 52.48% | 47.48% | **−5.00%** | `alexnet_experiment3.ipynb` |
| VGG-16 | 69.48% | 65.77% | **−3.70%** | `vgg16_experiment3.ipynb` |

---

## Key Finding

Results split cleanly along architectural lines:

- **ResNet-18 and ResNet-50** (residual / skip-connection architectures): accuracy maintained or improved after frequency filtering. The mask converges on a configuration that amplifies low-to-mid frequencies and mildly suppresses high frequencies, improving signal-to-noise ratio without discarding useful information.
- **AlexNet and VGG-16** (sequential architectures, no skip connections): accuracy degrades substantially. Frequency filtering removes components that these architectures depend on — they appear to exploit a broader, more uniform frequency spectrum.

**Hypothesis:** Residual connections enable frequency-selective processing as an architectural inductive bias. Skip connections allow the network to route information around frequency-distorted intermediate representations, making ResNets robust to and capable of benefiting from spectral filtering. Sequential architectures lack this routing, making them sensitive to any spectral modification.

This hypothesis motivates Phase 3: if the same frequency preference is recovered when a pretrained classifier jointly fine-tunes with the mask (Phase 3.2), the preference is architectural rather than a product of static inference.

---

## Artifacts

Each architecture's results are stored in `results/<arch>/`:

| File | Description |
|---|---|
| `learned_mask.pt` | Trained mask weights (PyTorch tensor) |
| `learned_mask.png` | Frequency mask visualisation (2D heatmap) |
| `training_history.pt` | Per-epoch loss and accuracy |
| `summary.txt` | Numerical results summary |

| Architecture | Results Directory |
|---|---|
| ResNet-18 | `results/resnet18/` |
| ResNet-50 | `results/resnet50/` |
| AlexNet | `results/alexnet/` |
| VGG-16 | `results/vgg16/` |
