"""
Learnable frequency masks for filtering in the frequency domain.
"""

import torch
import torch.nn as nn


class Learnable2DFrequencyMask(nn.Module):
    """
    Full 2D learnable mask - 224x224 parameters.
    Each frequency pixel has its own learnable weight.

    Activation modes:
        'sigmoid': mask = scale * sigmoid(weights). Bounded to (0, scale),
                   centered at scale/2. Non-negative, explosion-proof.
        'normalize': mask = weights / mean(weights). Mean forced to 1.0,
                     but range is unbounded — can explode with noisy gradients.
        None: raw weights, no constraint.
    """

    def __init__(self, image_size=224, init_value=1.0, init_std=0.1,
                 normalize=False, activation=None, sigmoid_scale=2.0):
        """
        Args:
            image_size: Size of the frequency domain mask (default: 224)
            init_value: Initial mean value for mask weights (default: 1.0)
            init_std: Standard deviation for weight initialization (default: 0.1)
            normalize: If True, normalize mask to mean=1.0 during forward pass.
                       Deprecated — prefer activation='sigmoid'.
            activation: Activation to apply to raw weights before masking.
                        'sigmoid': 2*sigmoid(w) — bounded (0, sigmoid_scale), non-negative.
                        None: no activation (use normalize for legacy behavior).
            sigmoid_scale: Max value for sigmoid activation (default: 2.0).
                           mask = sigmoid_scale * sigmoid(weights).
        """
        super(Learnable2DFrequencyMask, self).__init__()

        self.image_size = image_size
        self.normalize = normalize and activation is None  # normalize disabled when using activation
        self.activation = activation
        self.sigmoid_scale = sigmoid_scale

        if activation == 'sigmoid':
            # Initialize so sigmoid(w) ≈ init_value / sigmoid_scale
            # sigmoid(x) = init_value/scale => x = logit(init_value/scale)
            target = max(min(init_value / sigmoid_scale, 0.999), 0.001)
            logit_value = torch.log(torch.tensor(target / (1.0 - target)))
            initial_mask = torch.full((1, 1, image_size, image_size), logit_value.item())
            initial_mask += torch.randn(1, 1, image_size, image_size) * init_std
        else:
            initial_mask = torch.ones(1, 1, image_size, image_size) * init_value
            initial_mask += torch.randn(1, 1, image_size, image_size) * init_std

        self.mask_weights = nn.Parameter(initial_mask)

    def _apply_activation(self, raw):
        """Convert raw weights to the effective mask."""
        if self.activation == 'sigmoid':
            return self.sigmoid_scale * torch.sigmoid(raw)
        elif self.normalize:
            return raw / (raw.mean() + 1e-8)
        return raw

    def forward(self, fft_result):
        """
        Apply 2D mask to frequency domain.

        Args:
            fft_result: Complex tensor of shape [B, C, H, W]

        Returns:
            masked_fft: Masked frequency domain tensor [B, C, H, W]
        """
        mask = self._apply_activation(self.mask_weights)

        # Expand mask to match channels (apply same mask to R, G, B)
        mask_expanded = mask.expand(-1, fft_result.size(1), -1, -1)

        # Apply mask
        return fft_result * mask_expanded

    def get_mask_visualization(self):
        """
        Return the effective mask (after activation) as numpy array.

        Returns:
            mask_array: Numpy array of shape [H, W]
        """
        mask = self._apply_activation(self.mask_weights)
        return mask[0, 0].detach().cpu().numpy()
