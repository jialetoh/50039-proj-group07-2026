import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from src.utils import TRAIN_DIR, TEST_DIR, GROUND_TRUTH_DIR, get_device


def _collect_images(folder: Path):
    """Collect png image paths from one directory."""
    paths = folder.rglob("*.png")
    return sorted(paths)

def get_dl_kwargs():
    """Dynamically set DataLoader arguments based on device."""
    device = get_device()
    if device == "cuda":
        return {"num_workers": 4, "pin_memory": True}
    else:
        # Safe defaults for MPS (Apple Silicon) and CPU
        return {"num_workers": 0, "pin_memory": False}


# ==========================================
# Transforms
# ==========================================

def get_transforms(augment_mode=0, image_size=(256, 256)):
    """
    Helper to get transforms with different augmentation strategies.
    augment_mode:
        0 = None
        1 = Rotation, translation, brightness, contrast, saturation
    """
    base_transforms = [transforms.Resize(image_size)]
    
    match augment_mode:
        case 0:
            # Unaugmented
            pass
            
        case 1:
            # ±10º rotation, ±10% vertical/horizontal translation
            # ±10% brightness/contrast, ±5% saturation, 0 hue shift
            base_transforms.extend([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ColorJitter(
                    brightness=0.1, 
                    contrast=0.1, 
                    saturation=0.05, 
                    hue=0.0
                )
            ])
            
        case _:
            raise ValueError(f"Unknown augment_mode: {augment_mode}. Choose 0 or 1.")

    # Convert to tensor
    base_transforms.append(transforms.ToTensor())
    
    return transforms.Compose(base_transforms)


# ==========================================
# Dataset
# ==========================================

class CableTrainDataset(Dataset):
    """Loads normal cable images for training."""
    def __init__(self, train_dir: Path, transform=None):
        self.image_paths = _collect_images(train_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

class CableTestDataset(Dataset):
    """
    Loads test images with labels and corresponding ground truth masks.
    Normal: Label 0 + blank ground truth masks.
    Anomalous: Label 1 + corresponding ground truth masks.
    """
    def __init__(self, test_dir: Path, gt_dir: Path, image_size=(256, 256)):
        self.test_dir = test_dir
        self.gt_dir = gt_dir
        self.samples = []  # List of tuples: (img_path, mask_path, has_anomaly_int)
        
        # Transforms
        self.img_transform = get_transforms(augment_mode=0, image_size=image_size)
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        # Populate self.samples with paths and labels
        for sub_dir in sorted(self.test_dir.iterdir()):
            if not sub_dir.is_dir(): continue

            defect_category = sub_dir.name
            is_anomaly = 0 if defect_category == "good" else 1
            
            for img_path in _collect_images(sub_dir):
                mask_path = None
                
                # If anomalous, locate the corresponding ground truth mask
                if is_anomaly == 1:
                    mask_name = f"{img_path.stem}_mask.png"
                    mask_path = self.gt_dir / defect_category / mask_name
                    
                    if not mask_path.exists():
                        print(f"Warning: Missing mask for {img_path.name}")
                        
                self.samples.append((img_path, mask_path, is_anomaly))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, has_anomaly = self.samples[idx]
        
        # Load and transform the image
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)
        
        # Load the mask
        if mask_path and mask_path.exists():
            mask_img = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask_img)
        else:
            # Create blank mask for normal images
            _, h, w = img.shape
            mask = torch.zeros((1, h, w), dtype=torch.float32)

        return img, mask, has_anomaly


# ==========================================
# DataLoaders
# ==========================================

def get_train_val_dataloaders(
    batch_size=16,
    val_split=0.15,
    train_transform=None,
    image_size=(256,256),
    seed=42):
    """Returns train and val dataloaders."""
    val_transform = get_transforms(augment_mode=0, image_size=image_size)
    
    if train_transform is None:
        train_transform = val_transform

    full_train_dataset = CableTrainDataset(TRAIN_DIR, transform=train_transform)
    full_val_dataset = CableTrainDataset(TRAIN_DIR, transform=val_transform)

    dataset_size = len(full_train_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    # Train val split
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_val_dataset, val_indices)

    # Create dataloaders
    dl_kwargs = get_dl_kwargs()
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        **dl_kwargs
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        **dl_kwargs
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=16, image_size=(256, 256)):
    """Returns the test dataloader containing images and ground-truth masks."""
    dataset = CableTestDataset(
        test_dir=TEST_DIR, 
        gt_dir=GROUND_TRUTH_DIR,
        image_size=image_size
    )

    dl_kwargs = get_dl_kwargs()
    test_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        **dl_kwargs
    )

    return test_loader
