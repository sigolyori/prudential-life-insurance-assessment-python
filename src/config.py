from pathlib import Path
import random
import os
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Columns
TARGET_COL = "Response"
ID_COL = "Id"

# Reproducibility
RANDOM_STATE = 42
N_FOLDS = 5


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Seed python, numpy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
