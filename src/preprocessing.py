"""
Data Preprocessing Pipeline:
- Numerical feature identification & metadata separation (zero data leakage)
- Missing value imputation (Mean / Median)
- Zero-variance feature removal
- Optional log1p transformation
- StandardScaler feature scaling
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def detect_column_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Categorize columns into numeric features, potential ID columns,
    and metadata/target columns to prevent data leakage.
    """
    total_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in total_cols if c not in numeric_cols]
    
    # Identify potential sample ID column
    id_candidates = []
    for c in total_cols:
        col_lower = str(c).lower()
        if "id" in col_lower or "sample" in col_lower or "patient" in col_lower or "barcode" in col_lower:
            id_candidates.append(c)
        elif df[c].nunique() == len(df) and df[c].dtype == object:
            id_candidates.append(c)
            
    # Priority for ID column
    detected_id = id_candidates[0] if id_candidates else None
    
    # Metadata columns: categorical or string columns that are not ID
    metadata_cols = [c for c in non_numeric_cols if c != detected_id]
    
    return {
        "all_columns": total_cols,
        "numeric_features": numeric_cols,
        "non_numeric_columns": non_numeric_cols,
        "detected_id": detected_id,
        "metadata_columns": metadata_cols
    }

def handle_missing_values(
    df: pd.DataFrame,
    feature_cols: List[str],
    strategy: str = "median"
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Impute missing values in numeric feature columns using mean or median.
    Tracks before/after missingness and total imputed cells.
    """
    X_df = df[feature_cols].copy()
    missing_before = int(X_df.isnull().sum().sum())
    missing_by_col = X_df.isnull().sum().to_dict()
    
    if missing_before > 0:
        imputer = SimpleImputer(strategy=strategy)
        imputed_array = imputer.fit_transform(X_df)
        X_df = pd.DataFrame(imputed_array, columns=feature_cols, index=df.index)
        
    missing_after = int(X_df.isnull().sum().sum())
    
    df_clean = df.copy()
    df_clean[feature_cols] = X_df
    
    stats = {
        "missing_before": missing_before,
        "missing_after": missing_after,
        "imputed_cells": missing_before - missing_after
    }
    return df_clean, stats

def remove_zero_variance(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 1e-5
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Identify and remove constant / zero-variance features that provide no signal.
    """
    variances = df[feature_cols].var()
    retained = variances[variances > threshold].index.tolist()
    dropped = variances[variances <= threshold].index.tolist()
    
    return df, retained, dropped

def apply_log1p_transformation(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Apply log1p variance-stabilizing transformation if data is compatible (non-negative).
    Safely disables with an informative message if negative values exist.
    """
    min_val = df[feature_cols].min().min()
    if min_val < 0:
        return df, False, f"Negative values detected (min: {min_val:.4f}). log1p requires non-negative values."
        
    df_log = df.copy()
    df_log[feature_cols] = np.log1p(df_log[feature_cols])
    return df_log, True, "log1p transformation applied successfully."

def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, StandardScaler, Dict[str, float]]:
    """
    Standardize features using StandardScaler: X_scaled = (X - mean) / std.
    Returns scaled matrix, scaler instance, and before/after distribution stats.
    """
    X = df[feature_cols].values
    mean_before = float(np.mean(X))
    std_before = float(np.std(X))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    mean_after = float(np.mean(X_scaled))
    std_after = float(np.std(X_scaled))
    
    stats = {
        "mean_before": mean_before,
        "std_before": std_before,
        "mean_after": mean_after,
        "std_after": std_after,
        "original_dim": (df.shape[0], len(feature_cols)),
        "processed_dim": X_scaled.shape
    }
    return X_scaled, scaler, stats
