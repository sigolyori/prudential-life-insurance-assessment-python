from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROCESSED_DIR / "final_pipe.joblib"

TARGET_COL = "Response"
ID_COL = "Id"

NUM_CLASSES = 8
REJECT_CLASS = 1
DEFAULT_TOP_K = 10

Language = Literal["ko", "en"]
DEFAULT_LANGUAGE: Language = "ko"

RANDOM_STATE = 42
N_FOLDS = 5


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def decision_from_class(pred_class: int) -> Literal["approve", "reject"]:
    return "reject" if int(pred_class) == REJECT_CLASS else "approve"
