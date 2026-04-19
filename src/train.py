import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pytorch_msssim import ssim


class L1SSIMLoss(nn.Module):
    """
    Combined L1 and Structural Similarity Index (SSIM) Loss.
    L1 encourages pixel-wise accuracy, while SSIM preserves perceptual structure.
    """
    def __init__(self, l1_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.l1 = nn.L1Loss()

    def forward(self, reconstructed, target):
        # L1 Loss
        l1_loss = self.l1(reconstructed, target)
        
        # SSIM Loss: ssim() returns 1.0 for identical images. 
        # We subtract from 1.0 to turn it into a minimizable loss.
        # data_range=1.0 because our images are scaled between [0.0, 1.0]
        ssim_val = ssim(reconstructed, target, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val
        
        # Combine
        return (self.l1_weight * l1_loss) + ((1.0 - self.l1_weight) * ssim_loss)


def train_autoencoder(
    model, 
    train_loader, 
    val_loader, 
    device, 
    epochs=50, 
    lr=1e-3, 
    l1_weight=0.5,
    save_path="best_model.pth",
    patience=10  # Early stopping parameter
):
    """
    Training loop for autoencoder
    """
    # Initialize Loss and Optimizer
    criterion = L1SSIMLoss(l1_weight=l1_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Tracking
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0  # Tracker
    
    # Ensure save directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        # ==========================================
        # Training Phase
        # ==========================================
        model.train()
        train_loss_accum = 0.0
        
        # Using tqdm for progress bar
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for images in train_loop:
            images = images.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(images)
            
            # Autoencoders reconstruct their own input
            loss = criterion(reconstructed, images)
            
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item() * images.size(0)
            train_loop.set_postfix(loss=loss.item())
            
        epoch_train_loss = train_loss_accum / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # ==========================================
        # Validation Phase
        # ==========================================
        model.eval()
        val_loss_accum = 0.0
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for images in val_loop:
                images = images.to(device)
                reconstructed = model(images)
                
                loss = criterion(reconstructed, images)
                val_loss_accum += loss.item() * images.size(0)
                val_loop.set_postfix(loss=loss.item())
                
        epoch_val_loss = val_loss_accum / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        
        # ==========================================
        # Logging, Model Saving & Early Stopping
        # ==========================================
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f" -> Validation loss improved! Model saved to {save_path}")
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1
            print(f" -> No improvement for {epochs_no_improve}/{patience} epoch(s).")
            
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered! Training halted at epoch {epoch+1}.")
                break # Exit the loop early

    print("Training complete.")
    return history


def plot_loss_curves(history, title_suffix="", save=False, save_path="loss_curve.png"):
    """
    Plots standard and log-scaled loss curves, marking the early stopping point.
    """
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    
    # Find the epoch where validation loss was lowest (where the model was saved)
    best_epoch = np.argmin(val_loss) + 1 
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear Scale Plot
    axes[0].plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(epochs, val_loss, label='Val Loss', color='orange', linewidth=2)
    axes[0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Model (Epoch {best_epoch})')
    axes[0].set_title(f'Loss Curve {title_suffix}', fontsize=14)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss (L1 + SSIM)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Log Scale Plot
    axes[1].plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
    axes[1].plot(epochs, val_loss, label='Val Loss', color='orange', linewidth=2)
    axes[1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Model (Epoch {best_epoch})')
    axes[1].set_title(f'Log Loss Curve {title_suffix}', fontsize=14)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Log Loss', fontsize=12)
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curve plot saved to {save_path}")
        
    plt.show()
