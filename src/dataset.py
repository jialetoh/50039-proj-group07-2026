import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms

from utils import TRAIN_DIR, TEST_DIR, set_seed

# ImageNet normalization constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def _collect_images(folder):
    """Collect image paths from one directory using common image extensions."""
    paths = []
    for pattern in _IMG_EXTS:
        paths.extend(folder.glob(pattern))
    return sorted(paths)


# ==========================================
# 1. Dataset
# ==========================================

class CableDataset(Dataset):
    """Loads cable images from a list of paths and applies an optional transform."""

    def __init__(self, image_paths: list, transform=None):
        self.image_paths = image_paths
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
    Loads all test images (normal + anomalous) with binary labels.
    Label 0 = normal, 1 = anomalous.
    """

    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []  # list of (Path, int)

        if not TEST_DIR.exists():
            raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

        # Normal test images live in TEST_DIR/good/
        good_dir = TEST_DIR / "good"
        if good_dir.exists():
            for p in _collect_images(good_dir):
                self.samples.append((p, 0))

        # Every other subdirectory contains anomalous images
        for sub in sorted(TEST_DIR.iterdir()):
            if sub.is_dir() and sub.name != "good":
                for p in _collect_images(sub):
                    self.samples.append((p, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ==========================================
# 2. Transforms
# ==========================================

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05,
                               saturation=0.05, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ==========================================
# 3. DataLoader factory
# ==========================================

def get_dataloaders(batch_size: int = 16, val_split: float = 0.15, seed: int = 42):
    """
    Returns (train_loader, val_loader, test_loader).

    train_loader  -- augmented normal images
    val_loader    -- non-augmented normal images (held-out split)
    test_loader   -- all test images with (image, label) pairs; label 0=normal, 1=anomalous
    """
    set_seed(seed)

    # Collect all normal training images
    train_good_dir = TRAIN_DIR / "good"
    if not train_good_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_good_dir}")
    all_paths = _collect_images(train_good_dir)
    if not all_paths:
        raise RuntimeError(f"No training images found in: {train_good_dir}")

    n_total = len(all_paths)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    # Deterministic index split
    generator = torch.Generator().manual_seed(seed)
    indices = list(range(n_total))
    train_indices, val_indices = random_split(indices, [n_train, n_val],
                                              generator=generator)

    train_ds = torch.utils.data.Subset(
        CableDataset(all_paths, transform=get_train_transform()),
        list(train_indices),
    )
    val_ds = torch.utils.data.Subset(
        CableDataset(all_paths, transform=get_eval_transform()),
        list(val_indices),
    )
    test_ds = CableTestDataset(transform=get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader
