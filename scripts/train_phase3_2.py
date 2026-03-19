#!/usr/bin/env python3
"""Phase 3.2 training: Warmup + Progressive Unfreezing for frequency mask analysis.

Multi-GPU (DDP) and AMP support.

Usage:
  # Single GPU:
  uv run train-phase3 --arch resnet18 --data-dir /path/to/cache

  # Multi-GPU:
  uv run torchrun --nproc_per_node=4 scripts/train_phase3_2.py --arch resnet18 --data-dir /path/to/cache
"""

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
from torch.utils.data import random_split
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from frequency.transforms import apply_fft, apply_ifft
from frequency.mask import Learnable2DFrequencyMask
from scripts.distributed import (
    setup_distributed, cleanup_distributed,
    wrap_model, unwrap_model, make_dataloader, is_main,
)

# --- Hyperparameters (defaults, CNN-tuned) ---
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
    'vit_b_16': lambda: tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1),
}

# Per-architecture overrides (DeiT-style for ViT)
ARCH_HPARAMS = {
    'vit_b_16': {
        'optimizer': 'adamw',
        'weight_decay': 0.05,
        'classifier_lr': 0.00002,
    },
}


def get_hparams(arch):
    """Return merged hyperparameters for the given architecture."""
    defaults = {
        'optimizer': 'adam',
        'weight_decay': WEIGHT_DECAY,
        'classifier_lr': CLASSIFIER_LR,
        'mask_lr_warmup': MASK_LR_WARMUP,
        'mask_lr_finetune': MASK_LR_FINETUNE,
    }
    overrides = ARCH_HPARAMS.get(arch, {})
    return {**defaults, **overrides}


def make_optimizer(param_groups, hp):
    """Create Adam or AdamW based on arch hparams."""
    cls = optim.AdamW if hp['optimizer'] == 'adamw' else optim.Adam
    return cls(param_groups, weight_decay=hp['weight_decay'])


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

    def unfreeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = True

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


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.item())
        return res


def evaluate(pipeline, dataloader, criterion, device, use_amp):
    pipeline.eval()
    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs, _ = pipeline(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            c1, c5 = accuracy(outputs, labels, topk=(1, 5))
            total += labels.size(0)
            correct1 += c1
            correct5 += c5
    return total_loss / len(dataloader), 100.0 * correct1 / total, 100.0 * correct5 / total


def save_final_plots(history, baseline_acc1, results_dir, arch):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    epochs_range = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax in axes.flat:
        ax.axvspan(1, min(WARMUP_EPOCHS + 0.5, len(history['train_loss']) + 0.5),
                    alpha=0.08, color='blue', label='Warmup')

    axes[0, 0].plot(epochs_range, history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs_range, history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs_range, history['train_acc1'], label='Train', linewidth=2)
    axes[0, 1].plot(epochs_range, history['val_acc1'], label='Validation', linewidth=2)
    axes[0, 1].axhline(y=baseline_acc1, color='red', linestyle='--', label=f'Baseline ({baseline_acc1:.1f}%)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Top-1 Accuracy (%)')
    axes[0, 1].set_title('Top-1 Accuracy', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs_range, history['train_acc5'], label='Train', linewidth=2)
    axes[1, 0].plot(epochs_range, history['val_acc5'], label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-5 Accuracy (%)')
    axes[1, 0].set_title('Top-5 Accuracy', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_range, history['mask_std'], linewidth=2, color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_title('Mask Diversity (std)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'{arch} Phase 3.2: Warmup + Progressive Unfreezing', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "training_history.png", dpi=150, bbox_inches='tight')
    plt.close()

    mask_viz_path = results_dir / "learned_mask_viz.npy"
    if mask_viz_path.exists():
        mask_viz = np.load(mask_viz_path)
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
    parser.add_argument('--no-amp', action='store_true',
                        help="Disable automatic mixed precision")
    args = parser.parse_args()

    arch = args.arch
    rank, local_rank, world_size, device, is_distributed = setup_distributed()
    use_amp = not args.no_amp and device.type == "cuda"

    if is_main(rank):
        print(f"Device: {device} | World size: {world_size} | AMP: {use_amp}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(local_rank)}")

    results_dir = PROJECT_ROOT / "experiments" / "results" / f"{arch}_phase3.2"
    if is_main(rank):
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results: {results_dir}")

    # --- Data ---
    from datasets import load_from_disk
    from torchvision import transforms

    cache_path = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "imagenet_100k_cache"
    if not cache_path.exists():
        if is_main(rank):
            print(f"ERROR: Cache not found at {cache_path}")
        cleanup_distributed()
        sys.exit(1)

    if is_main(rank):
        print(f"Loading dataset from {cache_path}...")
    imagenet_subset = load_from_disk(str(cache_path))
    if is_main(rank):
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

    num_workers = min(16, os.cpu_count() or 1)
    dl_kwargs = dict(num_workers=num_workers, pin_memory=True, persistent_workers=True)

    train_loader, train_sampler = make_dataloader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, is_distributed=is_distributed, **dl_kwargs)
    val_loader, _ = make_dataloader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, is_distributed=is_distributed, **dl_kwargs)

    if is_main(rank):
        print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # --- Model ---
    if is_main(rank):
        print(f"\nLoading {arch} with pretrained ImageNet weights...")
    classifier = ARCH_LOADERS[arch]()
    pipeline_raw = JointTrainingPipeline(classifier, image_size=224)

    if is_main(rank):
        classifier_params = sum(p.numel() for p in pipeline_raw.classifier.parameters())
        mask_params = sum(p.numel() for p in pipeline_raw.freq_mask.parameters())
        print(f"  Classifier params: {classifier_params:,}")
        print(f"  Mask params: {mask_params:,}")

    pipeline = wrap_model(pipeline_raw, device, local_rank, is_distributed)
    raw_pipeline = unwrap_model(pipeline)

    # --- Baseline ---
    hp = get_hparams(arch)
    # Linear LR scaling: effective batch = BATCH_SIZE * world_size
    lr_scale = world_size
    mask_lr_warmup = hp['mask_lr_warmup'] * lr_scale
    mask_lr_finetune = hp['mask_lr_finetune'] * lr_scale
    classifier_lr = hp['classifier_lr'] * lr_scale
    if is_main(rank):
        print(f"\n  Optimizer: {hp['optimizer']} | Weight decay: {hp['weight_decay']}")
        if world_size > 1:
            print(f"  LR scaling: x{world_size} (mask warmup: {hp['mask_lr_warmup']}->{mask_lr_warmup}, "
                  f"mask finetune: {hp['mask_lr_finetune']}->{mask_lr_finetune}, classifier: {hp['classifier_lr']}->{classifier_lr})")

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if is_main(rank):
        print("\nEvaluating baseline (pretrained weights, mask=1.0)...")
    baseline_loss, baseline_acc1, baseline_acc5 = evaluate(pipeline, val_loader, criterion, device, use_amp)
    if is_main(rank):
        print(f"Baseline: Top1={baseline_acc1:.2f}%, Top5={baseline_acc5:.2f}% (loss={baseline_loss:.4f})")

    # --- Training state ---
    history = {
        'train_loss': [], 'train_acc1': [], 'train_acc5': [],
        'val_loss': [], 'val_acc1': [], 'val_acc5': [],
        'mask_std': [], 'stage': [],
    }
    best_val_acc = baseline_acc1
    patience_counter = 0
    start_epoch = 0
    in_finetune_stage = False

    checkpoint_path = results_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        if is_main(rank):
            print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_pipeline.load_state_dict(checkpoint['pipeline_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        patience_counter = checkpoint.get('patience_counter', 0)
        in_finetune_stage = checkpoint.get('in_finetune_stage', False)
        history = checkpoint['history']

        if in_finetune_stage:
            raw_pipeline.unfreeze_classifier()
            optimizer = make_optimizer([
                {'params': raw_pipeline.get_mask_params(), 'lr': mask_lr_finetune},
                {'params': raw_pipeline.get_classifier_params(), 'lr': classifier_lr},
            ], hp)
        else:
            raw_pipeline.freeze_classifier()
            optimizer = make_optimizer(
                [{'params': raw_pipeline.get_mask_params(), 'lr': mask_lr_warmup}], hp)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if is_main(rank):
            print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")
    else:
        raw_pipeline.freeze_classifier()
        if is_main(rank):
            print("  [Warmup] Classifier FROZEN")
        optimizer = make_optimizer(
            [{'params': raw_pipeline.get_mask_params(), 'lr': mask_lr_warmup}], hp)
        if is_main(rank):
            print("Starting fresh training")
            print(f"Baseline to beat: {baseline_acc1:.2f}%")

    # --- Training loop ---
    if is_main(rank):
        print(f"\nTraining for {EPOCHS} epochs (from {start_epoch})...")
        print("=" * 70)

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Stage transition: warmup -> fine-tune
        if epoch == WARMUP_EPOCHS and not in_finetune_stage:
            if is_main(rank):
                print(f"\n{'='*70}")
                print(f"STAGE TRANSITION at epoch {epoch+1}: Warmup -> Fine-tune")
                print(f"  Unfreezing classifier with LR={classifier_lr}")
                print(f"  Reducing mask LR: {mask_lr_warmup} -> {mask_lr_finetune}")
                print(f"{'='*70}\n")

            raw_pipeline.unfreeze_classifier()
            in_finetune_stage = True

            optimizer = make_optimizer([
                {'params': raw_pipeline.get_mask_params(), 'lr': mask_lr_finetune},
                {'params': raw_pipeline.get_classifier_params(), 'lr': classifier_lr},
            ], hp)

            patience_counter = 0

        # Train
        pipeline.train()
        train_loss = 0.0
        correct1 = 0
        correct5 = 0
        total = 0
        stage_label = "warmup" if not in_finetune_stage else "finetune"

        loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [{stage_label}]") if is_main(rank) else train_loader
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs, _ = pipeline(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            c1, c5 = accuracy(outputs, labels, topk=(1, 5))
            total += labels.size(0)
            correct1 += c1
            correct5 += c5

            if is_main(rank) and batch_idx % 20 == 0:
                loader.set_postfix({'loss': f"{loss.item():.3f}", 'top1': f"{100.*correct1/total:.1f}%"})

        # Validate
        val_loss, val_acc1, val_acc5 = evaluate(pipeline, val_loader, criterion, device, use_amp)

        train_loss /= len(train_loader)
        train_acc1 = 100.0 * correct1 / total
        train_acc5 = 100.0 * correct5 / total
        epoch_time = time.time() - epoch_start

        mask_viz = raw_pipeline.get_mask_visualization()

        history['train_loss'].append(train_loss)
        history['train_acc1'].append(train_acc1)
        history['train_acc5'].append(train_acc5)
        history['val_loss'].append(val_loss)
        history['val_acc1'].append(val_acc1)
        history['val_acc5'].append(val_acc5)
        history['mask_std'].append(float(mask_viz.std()))
        history['stage'].append(stage_label)

        if is_main(rank):
            print(f"\nEpoch {epoch+1}/{EPOCHS} [{stage_label}] ({epoch_time:.1f}s)")
            print(f"  Train: Loss={train_loss:.4f}, Top1={train_acc1:.2f}%, Top5={train_acc5:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Top1={val_acc1:.2f}%, Top5={val_acc5:.2f}%")
            print(f"  Mask (normalized): mean={mask_viz.mean():.3f}, std={mask_viz.std():.3f}")
            print(f"  Gap:   {train_acc1 - val_acc1:.2f}% (train - val)")
            print(f"  vs Baseline: {val_acc1 - baseline_acc1:+.2f}%")

        is_best = val_acc1 > best_val_acc
        if is_best:
            best_val_acc = val_acc1
            patience_counter = 0
            if is_main(rank):
                print(f"  ** New best: {best_val_acc:.2f}% (top5={val_acc5:.2f}%) **")
                torch.save(
                    {'pipeline_state_dict': raw_pipeline.state_dict(), 'val_acc1': val_acc1, 'val_acc5': val_acc5},
                    results_dir / "best_model.pt",
                )
        else:
            patience_counter += 1
            if is_main(rank):
                print(f"  No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")

        if is_main(rank):
            torch.save({
                'epoch': epoch + 1,
                'pipeline_state_dict': raw_pipeline.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'patience_counter': patience_counter,
                'in_finetune_stage': in_finetune_stage,
                'history': history,
                'baseline_acc1': baseline_acc1,
            }, checkpoint_path)

        if patience_counter >= EARLY_STOP_PATIENCE:
            if is_main(rank):
                print(f"\nEarly stopping at epoch {epoch+1}!")
            break

        if is_main(rank):
            print("-" * 70)

    # --- Save final artifacts (rank 0 only) ---
    if is_main(rank):
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print(f"Baseline Top1: {baseline_acc1:.2f}%")
        print(f"Best Validation Top1: {best_val_acc:.2f}%")
        print(f"Improvement: {best_val_acc - baseline_acc1:+.2f}%")
        print("=" * 70)

        best_path = results_dir / "best_model.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            raw_pipeline.load_state_dict(best_ckpt['pipeline_state_dict'])
            print(f"Loaded best model (Top1: {best_ckpt['val_acc1']:.2f}%, Top5: {best_ckpt['val_acc5']:.2f}%)")

        torch.save(raw_pipeline.freq_mask.state_dict(), results_dir / "learned_mask.pt")
        torch.save(history, results_dir / "training_history.pt")

        learned_mask = raw_pipeline.get_mask_visualization()
        np.save(results_dir / "learned_mask_viz.npy", learned_mask)

        with open(results_dir / "summary.txt", 'w') as f:
            f.write(f"{arch} Phase 3.2 - Warmup + Progressive Unfreezing\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Architecture: {arch} (ImageNet pretrained)\n")
            f.write(f"Train samples: {len(train_dataset):,}\n")
            f.write(f"Val samples: {len(val_dataset):,}\n")
            f.write(f"GPUs: {world_size} | AMP: {use_amp}\n")
            f.write(f"Epochs trained: {len(history['train_acc1'])}\n")
            f.write(f"Warmup epochs: {WARMUP_EPOCHS}\n")
            f.write(f"\nResults:\n")
            f.write(f"  Baseline Top1: {baseline_acc1:.2f}%\n")
            f.write(f"  Best Val Top1: {best_val_acc:.2f}%\n")
            f.write(f"  Improvement: {best_val_acc - baseline_acc1:+.2f}%\n")
            f.write(f"\nTwo-stage setup:\n")
            f.write(f"  Stage 1 (warmup): Classifier frozen, mask trains alone\n")
            f.write(f"  Stage 2 (finetune): Classifier unfrozen with LR={CLASSIFIER_LR}\n")
            f.write(f"\nHyperparameters:\n")
            f.write(f"  Warmup mask LR: {MASK_LR_WARMUP}\n")
            f.write(f"  Finetune mask LR: {MASK_LR_FINETUNE}\n")
            f.write(f"  Classifier LR: {CLASSIFIER_LR}\n")
            f.write(f"  Weight decay: {WEIGHT_DECAY}\n")
            f.write(f"  Batch size: {BATCH_SIZE} x {world_size} GPUs = {BATCH_SIZE * world_size} effective\n")
            f.write(f"  Data augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter\n")

        save_final_plots(history, baseline_acc1, results_dir, arch)
        print(f"\nAll artifacts saved to: {results_dir}")

    cleanup_distributed()


if __name__ == '__main__':
    main()
