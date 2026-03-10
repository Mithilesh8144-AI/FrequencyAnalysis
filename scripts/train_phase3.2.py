#!/usr/bin/env python3
"""Phase 3.2 training: Warmup + Progressive Unfreezing for frequency mask analysis."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from frequency.transforms import apply_fft, apply_ifft
from frequency.mask import Learnable2DFrequencyMask

# --- Hyperparameters (match notebook exactly) ---
WARMUP_EPOCHS = 10
EPOCHS = 60
MASK_LR_WARMUP = 0.005
MASK_LR_FINETUNE = 0.001
CLASSIFIER_LR = 0.00001
WEIGHT_DECAY = 0.0001
EARLY_STOP_PATIENCE = 15
BATCH_SIZE = 64
TRAIN_SPLIT = 0.8

ARCH_LOADERS = {
    'resnet18': lambda: tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1),
    'resnet50': lambda: tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1),
    'alexnet': lambda: tv_models.alexnet(weights=tv_models.AlexNet_Weights.IMAGENET1K_V1),
    'vgg16': lambda: tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1),
}


class JointTrainingPipeline(nn.Module):
    def __init__(self, classifier, image_size=224):
        super().__init__()
        self.classifier = classifier
        self.freq_mask = Learnable2DFrequencyMask(
            image_size=image_size,
            init_value=1.0,
            init_std=0.1,
            normalize=True,
        )

    def freeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = False
        print("  [Warmup] Classifier FROZEN")

    def unfreeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("  [Fine-tune] Classifier UNFROZEN")

    def forward(self, images):
        fft_result = apply_fft(images)
        masked_fft = self.freq_mask(fft_result)
        reconstructed = apply_ifft(masked_fft)
        outputs = self.classifier(reconstructed)
        return outputs, reconstructed

    def get_mask_params(self):
        return self.freq_mask.parameters()

    def get_classifier_params(self):
        return self.classifier.parameters()

    def get_mask_visualization(self):
        return self.freq_mask.get_mask_visualization()

    def get_raw_mask_stats(self):
        raw = self.freq_mask.mask_weights.detach()
        return {
            'mean': raw.mean().item(), 'std': raw.std().item(),
            'min': raw.min().item(), 'max': raw.max().item(),
        }


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, indices, transform):
        self.hf_dataset = hf_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.hf_dataset[real_idx]
        image = item['image']
        label = item['label']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def evaluate(pipeline, dataloader, criterion, device):
    pipeline.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = pipeline(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / len(dataloader), 100.0 * correct / total


def save_final_plots(history, baseline_acc, results_dir, arch):
    """Save training history plots at the end of training (no display)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    epochs_range = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax in axes:
        ax.axvspan(1, min(WARMUP_EPOCHS + 0.5, len(history['train_loss']) + 0.5),
                    alpha=0.08, color='blue', label='Warmup')

    axes[0].plot(epochs_range, history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(epochs_range, history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(epochs_range, history['val_acc'], label='Validation', linewidth=2)
    axes[1].axhline(y=baseline_acc, color='red', linestyle='--', label=f'Baseline ({baseline_acc:.1f}%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_range, history['mask_std'], linewidth=2, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Standard Deviation')
    axes[2].set_title('Mask Diversity (std)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'{arch} Phase 3.2: Warmup + Progressive Unfreezing', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "training_history.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save learned mask visualization
    mask_viz = np.load(results_dir / "learned_mask_viz.npy") if (results_dir / "learned_mask_viz.npy").exists() else None
    if mask_viz is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mask_viz, cmap='RdBu_r', vmin=0.5, vmax=1.5)
        ax.set_title(f'{arch} Phase 3.2 Learned Mask', fontweight='bold', fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Mask Weight')
        plt.savefig(results_dir / "learned_mask.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3.2: Warmup + Progressive Unfreezing")
    parser.add_argument('--arch', type=str, required=True, choices=list(ARCH_LOADERS.keys()),
                        help="Architecture to train")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Path to imagenet_100k_cache (default: data/imagenet_100k_cache)")
    args = parser.parse_args()

    arch = args.arch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = PROJECT_ROOT / "experiments" / "results" / f"{arch}_phase3.2"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {results_dir}")

    # --- Data ---
    from datasets import load_from_disk
    from torchvision import transforms

    cache_path = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "imagenet_100k_cache"
    if not cache_path.exists():
        print(f"ERROR: Cache not found at {cache_path}")
        sys.exit(1)

    print(f"Loading dataset from {cache_path}...")
    imagenet_subset = load_from_disk(str(cache_path))
    print(f"Loaded {len(imagenet_subset):,} images")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_size = int(TRAIN_SPLIT * len(imagenet_subset))
    val_size = len(imagenet_subset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(range(len(imagenet_subset)), [train_size, val_size], generator=generator)

    train_dataset = TransformSubset(imagenet_subset, train_indices.indices, train_transform)
    val_dataset = TransformSubset(imagenet_subset, val_indices.indices, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
    print(f"Train loader: {len(train_loader)} batches | Val loader: {len(val_loader)} batches")

    # --- Model ---
    print(f"\nLoading {arch} with pretrained ImageNet weights...")
    classifier = ARCH_LOADERS[arch]()
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"  Parameters: {classifier_params:,}")

    pipeline = JointTrainingPipeline(classifier, image_size=224)
    pipeline = pipeline.to(device)

    mask_params = sum(p.numel() for p in pipeline.freq_mask.parameters())
    print(f"  Mask params: {mask_params:,}")

    # --- Baseline ---
    criterion = nn.CrossEntropyLoss()
    print("\nEvaluating baseline (pretrained weights, mask=1.0)...")
    baseline_loss, baseline_acc = evaluate(pipeline, val_loader, criterion, device)
    print(f"Baseline: {baseline_acc:.2f}% (loss={baseline_loss:.4f})")

    # --- Training state ---
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'mask_std': [], 'stage': [],
    }
    best_val_acc = baseline_acc
    patience_counter = 0
    start_epoch = 0
    in_finetune_stage = False

    checkpoint_path = results_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        pipeline.load_state_dict(checkpoint['pipeline_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        patience_counter = checkpoint.get('patience_counter', 0)
        in_finetune_stage = checkpoint.get('in_finetune_stage', False)
        history = checkpoint['history']

        if in_finetune_stage:
            pipeline.unfreeze_classifier()
            optimizer = optim.Adam([
                {'params': pipeline.get_mask_params(), 'lr': MASK_LR_FINETUNE},
                {'params': pipeline.get_classifier_params(), 'lr': CLASSIFIER_LR},
            ], weight_decay=WEIGHT_DECAY)
        else:
            pipeline.freeze_classifier()
            optimizer = optim.Adam(
                pipeline.get_mask_params(), lr=MASK_LR_WARMUP,
                weight_decay=WEIGHT_DECAY,
            )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")
    else:
        pipeline.freeze_classifier()
        optimizer = optim.Adam(
            pipeline.get_mask_params(), lr=MASK_LR_WARMUP,
            weight_decay=WEIGHT_DECAY,
        )
        print("Starting fresh training")
        print(f"Baseline to beat: {baseline_acc:.2f}%")

    # --- Training loop ---
    print(f"\nTraining for {EPOCHS} epochs (from {start_epoch})...")
    print("=" * 70)

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()

        # Stage transition: warmup -> fine-tune
        if epoch == WARMUP_EPOCHS and not in_finetune_stage:
            print(f"\n{'='*70}")
            print(f"STAGE TRANSITION at epoch {epoch+1}: Warmup -> Fine-tune")
            print(f"  Unfreezing classifier with LR={CLASSIFIER_LR}")
            print(f"  Reducing mask LR: {MASK_LR_WARMUP} -> {MASK_LR_FINETUNE}")
            print(f"{'='*70}\n")

            pipeline.unfreeze_classifier()
            in_finetune_stage = True

            optimizer = optim.Adam([
                {'params': pipeline.get_mask_params(), 'lr': MASK_LR_FINETUNE},
                {'params': pipeline.get_classifier_params(), 'lr': CLASSIFIER_LR},
            ], weight_decay=WEIGHT_DECAY)

            patience_counter = 0
            print("  Patience counter reset to 0")

        # Train
        pipeline.train()
        train_loss = 0.0
        correct = 0
        total = 0
        stage_label = "warmup" if not in_finetune_stage else "finetune"

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [{stage_label}]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = pipeline(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 20 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.3f}", 'acc': f"{100.*correct/total:.1f}%"})

        # Validate
        val_loss, val_acc = evaluate(pipeline, val_loader, criterion, device)

        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total
        epoch_time = time.time() - epoch_start

        mask_viz = pipeline.get_mask_visualization()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['mask_std'].append(float(mask_viz.std()))
        history['stage'].append(stage_label)

        print(f"\nEpoch {epoch+1}/{EPOCHS} [{stage_label}] ({epoch_time:.1f}s)")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Mask (normalized): mean={mask_viz.mean():.3f}, std={mask_viz.std():.3f}")
        print(f"  Gap:   {train_acc - val_acc:.2f}% (train - val)")
        print(f"  vs Baseline: {val_acc - baseline_acc:+.2f}%")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"  ** New best: {best_val_acc:.2f}% **")
            torch.save(
                {'pipeline_state_dict': pipeline.state_dict(), 'val_acc': val_acc},
                results_dir / "best_model.pt",
            )
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")

        # Checkpoint every epoch
        torch.save({
            'epoch': epoch + 1,
            'pipeline_state_dict': pipeline.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'patience_counter': patience_counter,
            'in_finetune_stage': in_finetune_stage,
            'history': history,
            'baseline_acc': baseline_acc,
        }, checkpoint_path)

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}!")
            break

        print("-" * 70)

    # --- Save final artifacts ---
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Improvement: {best_val_acc - baseline_acc:+.2f}%")
    print("=" * 70)

    # Load best model for final saves
    best_path = results_dir / "best_model.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        pipeline.load_state_dict(best_ckpt['pipeline_state_dict'])
        print(f"Loaded best model (Val Acc: {best_ckpt['val_acc']:.2f}%)")

    # Save mask, history, visualization data
    torch.save(pipeline.freq_mask.state_dict(), results_dir / "learned_mask.pt")
    torch.save(history, results_dir / "training_history.pt")

    learned_mask = pipeline.get_mask_visualization()
    np.save(results_dir / "learned_mask_viz.npy", learned_mask)

    # Summary text
    with open(results_dir / "summary.txt", 'w') as f:
        f.write(f"{arch} Phase 3.2 - Warmup + Progressive Unfreezing\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Architecture: {arch} (ImageNet pretrained)\n")
        f.write(f"Train samples: {len(train_dataset):,}\n")
        f.write(f"Val samples: {len(val_dataset):,}\n")
        f.write(f"Epochs trained: {len(history['train_acc'])}\n")
        f.write(f"Warmup epochs: {WARMUP_EPOCHS}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Baseline Accuracy: {baseline_acc:.2f}%\n")
        f.write(f"  Best Val Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"  Improvement: {best_val_acc - baseline_acc:+.2f}%\n")
        f.write(f"\nTwo-stage setup:\n")
        f.write(f"  Stage 1 (warmup): Classifier frozen, mask trains alone\n")
        f.write(f"  Stage 2 (finetune): Classifier unfrozen with LR={CLASSIFIER_LR}\n")
        f.write(f"\nHyperparameters:\n")
        f.write(f"  Warmup mask LR: {MASK_LR_WARMUP}\n")
        f.write(f"  Finetune mask LR: {MASK_LR_FINETUNE}\n")
        f.write(f"  Classifier LR: {CLASSIFIER_LR}\n")
        f.write(f"  Weight decay: {WEIGHT_DECAY}\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n")
        f.write(f"  Data augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter\n")

    save_final_plots(history, baseline_acc, results_dir, arch)

    print(f"\nAll artifacts saved to: {results_dir}")


if __name__ == '__main__':
    main()
