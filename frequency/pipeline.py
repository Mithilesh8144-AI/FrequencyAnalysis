"""
Frequency filter pipeline combining FFT, mask, IFFT, and classifier.
"""

import torch.nn as nn
from .transforms import apply_fft, apply_ifft
from .mask import Learnable2DFrequencyMask


class FrequencyFilterPipeline(nn.Module):
    """
    Complete pipeline: Image -> FFT -> Learnable Mask -> IFFT -> Classifier

    Only the frequency mask is trainable!
    The classifier is frozen and provides gradients only.
    """

    def __init__(self, classifier, mask_config=None):
        """
        Args:
            classifier: Pre-trained classifier (will be frozen)
            mask_config: Dict with mask configuration:
                - image_size: Size of frequency mask (default: 224)
                - init_value: Initial mask value (default: 1.0)
                - init_std: Init std deviation (default: 0.1)
        """
        super(FrequencyFilterPipeline, self).__init__()

        # Frozen classifier
        self.classifier = classifier
        for param in self.classifier.parameters():
            param.requires_grad = False

        # Learnable 2D frequency mask (ONLY trainable component!)
        if mask_config is None:
            mask_config = {}

        self.freq_mask = Learnable2DFrequencyMask(
            image_size=mask_config.get('image_size', 224),
            init_value=mask_config.get('init_value', 1.0),
            init_std=mask_config.get('init_std', 0.1)
        )

        total_params = sum(p.numel() for p in self.freq_mask.parameters())
        print(f"\nFrequencyFilterPipeline created:")
        print(f"  Classifier: Frozen (0 trainable params)")
        print(f"  Frequency mask: {total_params:,} trainable params")

    def forward(self, images):
        """
        Apply frequency filtering pipeline.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            outputs: Classifier predictions [B, num_classes]
            reconstructed: Reconstructed images after frequency filtering [B, C, H, W]
        """
        # 1. Forward FFT (fixed transformation)
        fft_result = apply_fft(images)

        # 2. Apply learnable frequency mask
        masked_fft = self.freq_mask(fft_result)

        # 3. Inverse FFT (fixed transformation)
        reconstructed = apply_ifft(masked_fft)

        # 4. Forward through frozen classifier
        outputs = self.classifier(reconstructed)

        return outputs, reconstructed

    def get_trainable_params(self):
        """
        Get trainable parameters (only the frequency mask).

        Returns:
            parameters: Iterator of trainable parameters
        """
        return self.freq_mask.parameters()
