from pathlib import Path
import pandas as pd
from .config import DATA_DIR, TARGET_COL, ID_COL


def load_csvs(data_dir: Path | None = None):
    base = Path(data_dir) if data_dir else DATA_DIR
    train = pd.read_csv(base / "train.csv")
    test = pd.read_csv(base / "test.csv")
    sample = pd.read_csv(base / "sample_submission.csv")
    return train, test, sample


def split_X_y(df: pd.DataFrame, target_col: str = TARGET_COL):
    assert target_col in df.columns, f"Target column '{target_col}' not found."
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def load_train(data_dir: Path | None = None, drop_id: bool = True) -> pd.DataFrame:
    base = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_csv(base / "train.csv")
    if drop_id and ID_COL in df.columns:
        return df.drop(columns=[ID_COL])
    return df


def load_test(data_dir: Path | None = None, drop_id: bool = True) -> pd.DataFrame:
    base = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_csv(base / "test.csv")
    if drop_id and ID_COL in df.columns:
        return df.drop(columns=[ID_COL])
    return df


def load_sample_submission(data_dir: Path | None = None) -> pd.DataFrame:
    base = Path(data_dir) if data_dir else DATA_DIR
    return pd.read_csv(base / "sample_submission.csv")


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COL):
    return split_X_y(df, target_col)
