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
    """Sets the seed for reproducibility across the entire project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

