from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import DATA_DIR, ID_COL, TARGET_COL


def _base(data_dir: Path | None) -> Path:
    return Path(data_dir) if data_dir else DATA_DIR


def load_csvs(data_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _base(data_dir)
    train = pd.read_csv(base / "train.csv")
    test = pd.read_csv(base / "test.csv")
    sample = pd.read_csv(base / "sample_submission.csv")
    return train, test, sample


def load_train(data_dir: Path | None = None, drop_id: bool = True) -> pd.DataFrame:
    df = pd.read_csv(_base(data_dir) / "train.csv")
    if drop_id and ID_COL in df.columns:
        return df.drop(columns=[ID_COL])
    return df


def load_test(data_dir: Path | None = None, drop_id: bool = True) -> pd.DataFrame:
    df = pd.read_csv(_base(data_dir) / "test.csv")
    if drop_id and ID_COL in df.columns:
        return df.drop(columns=[ID_COL])
    return df


def load_sample_submission(data_dir: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(_base(data_dir) / "sample_submission.csv")


def split_X_y(df: pd.DataFrame, target_col: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.Series]:
    assert target_col in df.columns, f"Target column '{target_col}' not found."
    return df.drop(columns=[target_col]), df[target_col]


split_features_target = split_X_y


def binarize_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Collapse the 8-class ordinal Response into a binary underwriting label.

    Response == 1 → 1 (reject), Response ∈ {2..8} → 0 (approve).
    The Prudential dataset does not publish the precise semantics of the
    Response classes; this binarization encodes the assumption documented
    in the project README.
    """
    out = df.copy()
    out[target_col] = (out[target_col] == 1).astype(int)
    return out
