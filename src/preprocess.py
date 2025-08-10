from typing import Tuple, List, Dict, Union, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer

from .config import TARGET_COL, ID_COL


def infer_feature_types(df: pd.DataFrame, target_col: str = TARGET_COL, id_col: str = ID_COL) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c not in {target_col, id_col}]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    for_linear: bool = False,
    knn_impute_cols: Optional[List[str]] = None,
    mice_impute_cols: Optional[List[str]] = None,
    **imputer_kwargs
) -> ColumnTransformer:
    """
    Build a preprocessor with support for hybrid imputation strategies.
    
    Parameters:
    -----------
    numeric_cols : List[str]
        List of numeric column names
    categorical_cols : List[str]
        List of categorical column names
    for_linear : bool, default=False
        Whether to include scaling for linear models
    knn_impute_cols : List[str], optional
        Columns to use KNN imputation
    mice_impute_cols : List[str], optional
        Columns to use MICE (Iterative) imputation
    **imputer_kwargs : dict
        Additional arguments for imputers (e.g., n_neighbors for KNN)
    
    Returns:
    --------
    ColumnTransformer
        Configured preprocessor with specified imputation strategies
    """
    # Default imputation strategies
    if knn_impute_cols is None:
        knn_impute_cols = []
    if mice_impute_cols is None:
        mice_impute_cols = []
    
    # Get imputer parameters or use defaults
    n_neighbors = imputer_kwargs.get('n_neighbors', 5)
    random_state = imputer_kwargs.get('random_state', 0)
    
    # Define transformers for each imputation strategy
    transformers = []
    
    # 1. KNN Imputation for specified columns
    if knn_impute_cols:
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        transformers.append(('knn_impute', knn_imputer, knn_impute_cols))
    
    # 2. MICE (Iterative) Imputation for specified columns
    if mice_impute_cols:
        mice_imputer = IterativeImputer(random_state=random_state)
        transformers.append(('mice_impute', mice_imputer, mice_impute_cols))
    
    # 3. Standard numeric imputation (median) for remaining numeric columns
    remaining_num_cols = [
        col for col in numeric_cols 
        if col not in knn_impute_cols and col not in mice_impute_cols
    ]
    
    if remaining_num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if for_linear:
            num_steps.append(("scaler", StandardScaler(with_mean=True)))
            # Optionally add PowerTransformer for non-Gaussian features
            # num_steps.append(("power", PowerTransformer(method="yeo-johnson")))
        
        from sklearn.pipeline import Pipeline
        transformers.append(("num", Pipeline(num_steps), remaining_num_cols))
    
    # 4. Categorical pipeline (always uses most_frequent imputation)
    if categorical_cols:
        cat_pipeline = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
        from sklearn.pipeline import Pipeline
        transformers.append(("cat", Pipeline(cat_pipeline), categorical_cols))
    
    return ColumnTransformer(transformers, remainder="drop")


def build_preprocessor_from_df(
    df: pd.DataFrame, 
    for_linear: bool = False,
    knn_impute_cols: Optional[List[str]] = None,
    mice_impute_cols: Optional[List[str]] = None,
    **imputer_kwargs
) -> ColumnTransformer:
    """
    Build a preprocessor from a DataFrame with hybrid imputation support.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to infer feature types from
    for_linear : bool, default=False
        Whether to include scaling for linear models
    knn_impute_cols : List[str], optional
        Columns to use KNN imputation
    mice_impute_cols : List[str], optional
        Columns to use MICE (Iterative) imputation
    **imputer_kwargs : dict
        Additional arguments for imputers
        
    Returns:
    --------
    ColumnTransformer
        Configured preprocessor with specified imputation strategies
    """
    num_cols, cat_cols = infer_feature_types(df)
    return build_preprocessor(
        num_cols, 
        cat_cols, 
        for_linear=for_linear,
        knn_impute_cols=knn_impute_cols,
        mice_impute_cols=mice_impute_cols,
        **imputer_kwargs
    )
