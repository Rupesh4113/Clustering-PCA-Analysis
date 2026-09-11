"""
Principal Component Analysis (PCA) Module:
- Fit PCA on standardized feature matrix
- Coordinate extraction (2D and 3D)
- Individual and cumulative explained variance calculation
- Feature loadings analysis
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def run_pca(
    X_scaled: np.ndarray,
    n_components: int = 3,
    calibrate_scale: bool = True
) -> Tuple[np.ndarray, PCA, Dict[str, Any]]:
    """
    Fit PCA on standardized features and extract component coordinates.
    
    When calibrate_scale is True, scales coordinates to align with 3D geometry,
    ensuring standard DBSCAN eps (0.5-3.0) and K-Means inertia behavior
    regardless of input feature dimensionality (e.g. 500 features).
    """
    n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca_raw = pca.fit_transform(X_scaled)
    
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    
    if calibrate_scale and len(pca.explained_variance_) > 0:
        # Scale factor so mean component variance is calibrated to 3D embedding
        scale_factor = np.sqrt(np.mean(pca.explained_variance_)) / 2.75
        X_pca = X_pca_raw / max(scale_factor, 1e-6)
    else:
        X_pca = X_pca_raw
        
    stats = {
        "var_ratios": var_ratio.tolist(),
        "cumulative_variance": cum_var.tolist(),
        "total_variance_explained": float(cum_var[-1]),
        "pc1_variance": float(var_ratio[0]) if len(var_ratio) > 0 else 0.0,
        "pc2_variance": float(var_ratio[1]) if len(var_ratio) > 1 else 0.0,
        "pc3_variance": float(var_ratio[2]) if len(var_ratio) > 2 else 0.0,
        "singular_values": pca.singular_values_.tolist()
    }
    return X_pca, pca, stats

def get_top_loadings(
    pca_model: PCA,
    feature_names: List[str],
    component_idx: int = 0,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Retrieve top contributing features with highest absolute loadings for a given PC.
    """
    loadings = pca_model.components_[component_idx]
    loading_df = pd.DataFrame({
        "Feature": feature_names,
        "Loading": loadings,
        "Absolute_Loading": np.abs(loadings)
    }).sort_values(by="Absolute_Loading", ascending=False).head(top_n)
    
    return loading_df.reset_index(drop=True)
