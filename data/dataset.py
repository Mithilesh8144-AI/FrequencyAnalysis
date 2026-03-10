"""
Dataset module for loading ImageNet validation data from HuggingFace.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms


class HFImageNetDataset(Dataset):
    """Custom Dataset wrapper for Hugging Face ImageNet dataset."""

    def __init__(self, hf_dataset, transform=None):
        """
        Args:
            hf_dataset: HuggingFace dataset object
            transform: Optional torchvision transforms
        """
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, label


def load_imagenet_validation(subset_size=5000, image_size=224, batch_size=64):
    """
    Load ImageNet validation dataset from HuggingFace.

    Args:
        subset_size: Number of images to load (default: 5000)
        image_size: Image size for resizing (default: 224)
        batch_size: Batch size for DataLoader (default: 64)

    Returns:
        dataset: HFImageNetDataset instance
        dataloader: torch DataLoader
    """
    print(f"Loading ImageNet VALIDATION subset ({subset_size} images) using streaming...")
    print("This will only download the required samples, not the entire validation set.\n")

    # Load as streaming dataset to avoid downloading the entire split
    streaming_hf_dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        streaming=True
    )

    # Take first subset_size samples
    subset = list(streaming_hf_dataset.take(subset_size))

    print(f"✓ Downloaded {len(subset)} images")

    # Define ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create dataset
    dataset = HFImageNetDataset(subset, transform=transform)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"✓ Dataset created: {len(dataset)} images")
    print(f"✓ DataLoader created: batch_size={batch_size}, {len(dataloader)} batches")

    return dataset, dataloader


def get_denormalize_transform():
    """
    Returns a function to denormalize ImageNet images for visualization.

    Returns:
        denormalize: Function that takes a normalized tensor and returns denormalized tensor
    """
    def denormalize(img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        return torch.clamp(img, 0, 1)

    return denormalize
