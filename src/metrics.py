import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve
from skimage import measure
from pathlib import Path

def calculate_image_auroc(anomaly_maps, labels):
    """
    Calculates Image-level AUROC from pixel-wise anomaly maps.
    Uses maximum anomaly score in each map as the image-level score.
    """
    # Flatten each anomaly map into 1D vector
    # Use max pixel value in each map as image-level score
    image_scores = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(axis=1)
    return roc_auc_score(labels, image_scores)


def calculate_pixel_auroc(anomaly_maps, masks):
    """
    Calculates the Pixel-level AUROC from anomaly maps and binary GT masks.
    Treats each pixel as a binary classification.
    """
    # Flatten all pixels across all images
    flat_maps = anomaly_maps.flatten()
    flat_masks = masks.flatten()
    
    # Convert mask values to binary labels (0 or 1)
    flat_masks = (flat_masks > 0).astype(int)
    
    return roc_auc_score(flat_masks, flat_maps)


def calculate_aupro(anomaly_maps, masks, num_thresholds=100):
    """
    Calculates the Area Under the Per-Region Overlap (AUPRO) curve from 
    anomaly maps and GT masks. Groups ground truth pixels into connected 
    regions and evaluates the True Positive Rate (TPR) or per-region recall 
    at each threshold, preventing large defects from dominating the score.
    """
    # Flatten anomaly maps and masks to compute global False Positive Rate (FPR) later
    flat_maps = anomaly_maps.flatten()
    flat_masks = (masks.flatten() > 0).astype(int)
    
    # Get thresholds based on the min/max values of the anomaly maps
    min_val, max_val = anomaly_maps.min(), anomaly_maps.max()
    thresholds = np.linspace(min_val, max_val, num_thresholds)
    
    # Find connected regions in each GT mask
    # Each region is assigned a unique integer ID with measure.label
    labeled_masks = [measure.label(mask.squeeze()) for mask in masks]
    
    tpr_per_threshold = []
    fpr_per_threshold = []
    
    for thresh in thresholds:
        # Convert anomaly scores to binary predictions based on threshold
        binary_maps = (anomaly_maps > thresh).astype(int)
        
        region_tprs = []
        for i in range(len(masks)):
            pred_mask = binary_maps[i].squeeze()
            labeled_mask = labeled_masks[i]
            
            # Extract properties of each distinct defect region to compute TPR for each region
            regions = measure.regionprops(labeled_mask)
            for region in regions:
                # Create a mask specifically for this single region
                region_mask = (labeled_mask == region.label)
                
                # Calculate what percentage of THIS region the model successfully detected
                overlap = np.logical_and(region_mask, pred_mask).sum()
                region_tpr = overlap / region.area
                region_tprs.append(region_tpr)
        
        # Average TPR across all distinct regions
        mean_region_tpr = np.mean(region_tprs) if region_tprs else 0.0
        tpr_per_threshold.append(mean_region_tpr)
        
        # Calculate global False Positive Rate (FPR)
        # FPR = FP / (FP + TN)
        fp = np.logical_and(flat_maps > thresh, flat_masks == 0).sum()
        tn = np.logical_and(flat_maps <= thresh, flat_masks == 0).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_per_threshold.append(fpr)
        
    # Sort points by FPR to compute the Area Under the Curve properly
    fpr_per_threshold = np.array(fpr_per_threshold)
    tpr_per_threshold = np.array(tpr_per_threshold)
    
    sort_idx = np.argsort(fpr_per_threshold)
    fpr_sorted = fpr_per_threshold[sort_idx]
    tpr_sorted = tpr_per_threshold[sort_idx]
    
    # Integrate only up to FPR <= 0.3
    limit_idx = np.searchsorted(fpr_sorted, 0.3, side='right')
    
    if limit_idx == 0:
        return 0.0
        
    fpr_limited = fpr_sorted[:limit_idx]
    tpr_limited = tpr_sorted[:limit_idx]
    
    # Normalize the area calculation by the 0.3 limit to get the score
    aupro_score = auc(fpr_limited, tpr_limited) / 0.3
    return aupro_score


def plot_roc_curves(anomaly_maps, masks, labels, save=False, save_path="roc_curves.png"):
    """
    Plots the Image-level and Pixel-level ROC curves side-by-side.
    """
    # Image-level ROC calculation
    image_scores = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(axis=1)
    fpr_img, tpr_img, _ = roc_curve(labels, image_scores)
    roc_auc_img = auc(fpr_img, tpr_img)

    # Pixel-level ROC calculation
    flat_maps = anomaly_maps.flatten()
    flat_masks = (masks.flatten() > 0).astype(int)
    fpr_pix, tpr_pix, _ = roc_curve(flat_masks, flat_maps)
    roc_auc_pix = auc(fpr_pix, tpr_pix)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Image ROC
    axes[0].plot(fpr_img, tpr_img, color='darkorange', lw=2, label=f'Image ROC (AUC = {roc_auc_img:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('Image-level ROC Curve', fontsize=14)
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Plot Pixel ROC
    axes[1].plot(fpr_pix, tpr_pix, color='darkorange', lw=2, label=f'Pixel ROC (AUC = {roc_auc_pix:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('Pixel-level ROC Curve', fontsize=14)
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    plt.suptitle('Receiver Operating Characteristic (ROC) Curves', fontsize=18)
    plt.tight_layout()

    if save:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves plot saved to {save_path}")

    plt.show()

