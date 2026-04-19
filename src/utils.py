import random
import numpy as np
import torch
from pathlib import Path

# ==========================================
# 1. Project Path Definitions
# ==========================================
# Find project root based on where utils.py is located
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data" / "cable"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

# Output Paths
OUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


# ==========================================
# 2. Helper Functions
# ==========================================
def set_seed(seed: int = 42):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Device-specific seed setting
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic algorithms on CUDA
        torch.cudnn.deterministic = True
        torch.cudnn.benchmark = False
    elif torch.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    """Check for CUDA or MPS for Apple Silicon."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    return device


def tensor_to_img(t: torch.Tensor):
    """
    Convert PyTorch image tensor (C, H, W) to NumPy array (H, W, C)
    and within [0,1] for matplotlib display
    """
    img = t.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)