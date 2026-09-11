"""
Model Evaluation & Stability Analysis Module:
- Metrics computation (Silhouette, Davies-Bouldin)
- 50-Run Monte Carlo stability benchmark for K-Means++
- Stability classification (Very High, High, Moderate, Low)
"""

import time
from typing import Dict, Any, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    inertia: Optional[float] = None
) -> Dict[str, Any]:
    """Safely compute clustering evaluation metrics."""
    unique_labels = set(labels)
    cluster_labels = [l for l in unique_labels if l != -1]
    n_clusters = len(cluster_labels)
    
    # Evaluate only clustered points if noise exists
    valid_mask = labels != -1 if -1 in unique_labels else np.ones(len(labels), dtype=bool)
    n_valid = int(np.sum(valid_mask))
    
    sil = None
    db = None
    if n_clusters >= 2 and n_valid > n_clusters:
        try:
            sil = float(silhouette_score(X[valid_mask], labels[valid_mask]))
            db = float(davies_bouldin_score(X[valid_mask], labels[valid_mask]))
        except Exception:
            sil = None
            db = None
            
    return {
        "n_clusters": n_clusters,
        "silhouette": sil,
        "davies_bouldin": db,
        "inertia": inertia
    }

def run_stability_benchmark(
    X: np.ndarray,
    k: int = 4,
    n_runs: int = 50,
    base_seed: int = 42,
    progress_callback: Optional[Callable[[float, int], None]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    True Monte Carlo 50-run stability benchmark for K-Means++.
    Runs K-Means++ across genuinely distinct random seeds to quantify
    initialization sensitivity and cluster stability.
    """
    records = []
    
    for run_idx in range(n_runs):
        seed = base_seed + run_idx * 17  # Genuinely distinct seeds
        t0 = time.time()
        
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=seed
        )
        labels = km.fit_predict(X)
        elapsed = time.time() - t0
        
        sil = float(silhouette_score(X, labels))
        inertia = float(km.inertia_)
        
        cluster_counts = pd.Series(labels).value_counts().sort_index().to_dict()
        
        records.append({
            "run": run_idx + 1,
            "seed": seed,
            "silhouette": sil,
            "inertia": inertia,
            "execution_time": elapsed,
            "cluster_distribution": str(cluster_counts)
        })
        
        if progress_callback is not None:
            progress_callback((run_idx + 1) / n_runs, run_idx + 1)
            
    df_runs = pd.DataFrame(records)
    
    # Statistical synthesis
    mean_sil = float(df_runs["silhouette"].mean())
    std_sil = float(df_runs["silhouette"].std())
    min_sil = float(df_runs["silhouette"].min())
    max_sil = float(df_runs["silhouette"].max())
    
    mean_inertia = float(df_runs["inertia"].mean())
    std_inertia = float(df_runs["inertia"].std())
    
    # Stability Classification
    if std_sil < 0.010:
        stability_class = "Very High Stability"
        stability_badge = "🟢 Very High"
    elif std_sil < 0.025:
        stability_class = "High Stability"
        stability_badge = "🔵 High"
    elif std_sil < 0.050:
        stability_class = "Moderate Stability"
        stability_badge = "🟡 Moderate"
    else:
        stability_class = "Low Stability"
        stability_badge = "🔴 Low"
        
    summary = {
        "n_runs": n_runs,
        "k": k,
        "mean_silhouette": mean_sil,
        "std_silhouette": std_sil,
        "min_silhouette": min_sil,
        "max_silhouette": max_sil,
        "mean_inertia": mean_inertia,
        "std_inertia": std_inertia,
        "stability_class": stability_class,
        "stability_badge": stability_badge
    }
    
    return df_runs, summary
