import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from src.utils import tensor_to_img


def generate_anomaly_maps(model, dataloader, device):
    """
    Runs inference on the dataloader and generates pixel-wise anomaly maps.
    Returns the maps, ground truth masks, and image-level labels as NumPy arrays.
    """
    model.eval()
    all_maps = []
    all_masks = []
    all_labels = []
    
    loop = tqdm(dataloader, desc="Generating Anomaly Maps", leave=False)
    with torch.no_grad():
        for images, masks, labels in loop:
            images = images.to(device)
            reconstructed = model(images)
            
            # Calculate per-pixel error (L1 difference)
            error_map = torch.abs(images - reconstructed)
            
            # Average across 3 color channels to get a single grayscale anomaly map
            # [B, 3, H, W] -> [B, 1, H, W]
            anomaly_map = torch.mean(error_map, dim=1, keepdim=True)
            
            all_maps.append(anomaly_map.cpu().numpy())
            all_masks.append(masks.numpy())
            all_labels.append(labels.numpy())
            
    # Concatenate all batches into single arrays
    # Maps, ground truth masks, image-level labels
    return np.concatenate(all_maps), np.concatenate(all_masks), np.concatenate(all_labels)


def plot_anomaly_maps(model, dataset, anomaly_maps, device, save=False, save_path="anomaly_maps.png"):
    """
    For each class in dataset, plots five columns:
    Original image, reconstructed image, anomaly heatmap, GT mask, GT overlay.
    """
    # 1. Dynamically find one index for each category
    class_samples = {}
    for i, (img_path, _, _) in enumerate(dataset.samples):
        category_name = Path(img_path).parent.name
        if category_name not in class_samples:
            class_samples[category_name] = i
            
        # 8 defect classes + 1 normal class = 9 classes
        if len(class_samples) == 9: 
            break

    num_samples = len(class_samples)
    fig, axes = plt.subplots(num_samples, 5, figsize=(21, 4 * num_samples))

    for row, (category, idx) in enumerate(class_samples.items()):
        # Retrieve data
        img, gt_mask, _ = dataset[idx]

        img_np = tensor_to_img(img)        
        img_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            recon_tensor = model(img_tensor)
        recon_np = tensor_to_img(recon_tensor.squeeze(0).cpu())
        
        anom_map = anomaly_maps[idx].squeeze()
        gt_mask_np = gt_mask.squeeze().numpy()
        
        # --- Column 0: Original Image---
        ax_orig = axes[row, 0]
        ax_orig.imshow(img_np)
        ax_orig.axis("off")
        
        clean_category_name = category.replace('_', '\n').title()
        ax_orig.text(-0.05, 0.5, clean_category_name, fontsize=14,
                     ha='right', va='center', transform=ax_orig.transAxes)
        if row == 0: ax_orig.set_title("Original", fontsize=14, pad=10)
        
        # --- Column 1: Reconstructed Image---
        ax_recon = axes[row, 1]
        ax_recon.imshow(recon_np)
        ax_recon.axis("off")
        if row == 0: ax_recon.set_title("Reconstructed", fontsize=14, pad=10)
        
        # --- Column 2: Anomaly Map Heatmap ---
        ax_amap = axes[row, 2]
        im = ax_amap.imshow(anom_map, cmap='jet')
        ax_amap.axis("off")
        fig.colorbar(im, ax=ax_amap, fraction=0.046, pad=0.04)
        if row == 0: ax_amap.set_title("Anomaly Map (L1 Error)", fontsize=14, pad=10)
        
        # --- Column 3: Ground Truth Mask ---
        ax_gt = axes[row, 3]
        ax_gt.imshow(gt_mask_np, cmap='gray')
        ax_gt.axis("off")
        if row == 0: ax_gt.set_title("Ground Truth Mask", fontsize=14, pad=10)
        
        # --- Column 4: Overlay (Contour) ---
        ax_over = axes[row, 4]
        ax_over.imshow(anom_map, cmap='jet')
        if gt_mask_np.max() > 0:
            ax_over.contour(gt_mask_np, levels=[0.5], colors='white', linewidths=2)
        ax_over.axis("off")
        if row == 0: ax_over.set_title("Overlay (GT Contour on Map)", fontsize=14, pad=10)

    fig.suptitle('Test Set: Anomaly Maps vs Ground Truth by Class', fontsize=24, y=1.00)
    plt.tight_layout()
    
    if save:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Anomaly map heatmaps saved to {save_path}")
        
    plt.show()


