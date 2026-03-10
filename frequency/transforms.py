"""
Frequency domain transformations (FFT and IFFT).
"""

import torch
import torch.fft


def apply_fft(images):
    """
    Apply 2D FFT to a batch of images.

    Args:
        images: Tensor of shape [B, C, H, W]

    Returns:
        fft_result: Complex tensor of shape [B, C, H, W]
    """
    # Apply FFT to each channel separately
    fft_result = torch.fft.fft2(images, dim=(-2, -1))

    # Shift zero frequency to center
    fft_result = torch.fft.fftshift(fft_result, dim=(-2, -1))

    return fft_result


def apply_ifft(fft_result):
    """
    Apply inverse 2D FFT to reconstruct image.

    Args:
        fft_result: Complex tensor of shape [B, C, H, W]

    Returns:
        reconstructed: Real tensor of shape [B, C, H, W]
    """
    # Shift back from center
    fft_result = torch.fft.ifftshift(fft_result, dim=(-2, -1))

    # Apply inverse FFT
    reconstructed = torch.fft.ifft2(fft_result, dim=(-2, -1))

    # Take real part (imaginary should be ~0)
    reconstructed = reconstructed.real

    return reconstructed
