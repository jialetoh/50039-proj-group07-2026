import cv2
import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

SAT_THRESHOLD = 0.15
CLOSING_KSIZE = 7
DILATION_KSIZE = 5


def generate_cable_mask(
    imgs: torch.Tensor,
    sat_threshold: float = SAT_THRESHOLD,
    closing_ksize: int = CLOSING_KSIZE,
    dilation_ksize: int = DILATION_KSIZE,
) -> torch.Tensor:
    """Generate a binary cable mask from an RGB image tensor.

    Uses HSV saturation thresholding: the gray background has near-zero
    saturation while the cable (blue wire, copper strands) has higher
    saturation. Morphological closing fills internal holes; dilation adds
    a small safety margin around cable edges.

    Args:
        imgs: Float tensor [B, 3, H, W] in [0, 1].
        sat_threshold: Pixels with saturation > this value are cable.
        closing_ksize: Kernel size for morphological closing (odd int).
        dilation_ksize: Kernel size for morphological dilation (odd int).

    Returns:
        Float tensor [B, 1, H, W] with 1.0 = cable, 0.0 = background,
        on the same device as `imgs`.
    """
    device = imgs.device

    # --- HSV saturation (pure PyTorch, device-agnostic) ---
    max_rgb = imgs.max(dim=1, keepdim=True).values   # [B, 1, H, W]
    min_rgb = imgs.min(dim=1, keepdim=True).values   # [B, 1, H, W]
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)  # [B, 1, H, W]

    binary = (saturation > sat_threshold).squeeze(1).cpu().numpy().astype(np.uint8)
    # binary: [B, H, W] uint8

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_ksize, closing_ksize)
    )
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_ksize, dilation_ksize)
    )

    masks = []
    for b in range(binary.shape[0]):
        m = cv2.morphologyEx(binary[b], cv2.MORPH_CLOSE, close_kernel)
        m = cv2.dilate(m, dilate_kernel)
        masks.append(m)

    mask_np = np.stack(masks, axis=0)[:, np.newaxis, :, :]  # [B, 1, H, W]
    return torch.from_numpy(mask_np).float().to(device)


def masked_mae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MAE loss computed only over cable pixels.

    Divides by the actual number of cable pixels (× channels) rather than
    the full image size, giving the true mean error on the cable region.

    Args:
        recon:  [B, 3, H, W] reconstructed tensor.
        target: [B, 3, H, W] target tensor.
        mask:   [B, 1, H, W] binary cable mask (0/1 float).

    Returns:
        Scalar loss tensor.
    """
    pixel_errors = torch.abs(recon - target) * mask  # broadcast over channels
    n_cable_pixels = mask.sum() * recon.shape[1]     # sum over batch × channels
    return pixel_errors.sum() / (n_cable_pixels + 1e-8)


def masked_ssim_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """1 - SSIM loss computed on masked (background-zeroed) tensors.

    Args:
        recon:  [B, 3, H, W] reconstructed tensor.
        target: [B, 3, H, W] target tensor.
        mask:   [B, 1, H, W] binary cable mask (0/1 float).

    Returns:
        Scalar loss tensor.
    """
    masked_recon = recon * mask
    masked_target = target * mask
    ssim_fn = StructuralSimilarityIndexMeasure(
        data_range=1.0, kernel_size=11
    ).to(masked_recon.device)
    return 1.0 - ssim_fn(masked_recon, masked_target)


def compute_masked_error_map(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    """Per-pixel MAE error map with background zeroed out.

    Args:
        recon:  [1, 3, H, W] or [3, H, W] reconstructed tensor.
        target: same shape as recon.
        mask:   [1, 1, H, W] or [1, H, W] binary cable mask.

    Returns:
        [H, W] numpy float32 array; background pixels are 0.
    """
    if recon.dim() == 3:
        recon = recon.unsqueeze(0)
        target = target.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)

    with torch.no_grad():
        err = torch.abs(recon - target).mean(dim=1, keepdim=True)  # [1, 1, H, W]
        err = err * mask
    return err.squeeze().cpu().numpy()
