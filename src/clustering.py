"""
Clustering Algorithms Module:
- K-Means (random initialization)
- K-Means++ (probabilistic centroid initialization)
- K range parameter sweep (Silhouette & Davies-Bouldin evaluation)
- DBSCAN density-based clustering with noise identification
"""

import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

def run_kmeans(
    X: np.ndarray,
    k: int = 4,
    init: str = "random",
    n_init: int = 20,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray, Dict[str, Any]]:
    """Execute K-Means with specified initialization and parameters."""
    t0 = time.time()
    model = KMeans(
        n_clusters=k,
        init=init,
        n_init=n_init,
        random_state=random_state
    )
    labels = model.fit_predict(X)
    elapsed = time.time() - t0
    
    metrics = {
        "k": k,
        "inertia": float(model.inertia_),
        "init": init,
        "execution_time_sec": elapsed,
        "cluster_counts": pd.Series(labels).value_counts().sort_index().to_dict()
    }
    return model, labels, metrics

def run_kmeans_plus_plus(
    X: np.ndarray,
    k: int = 4,
    n_init: int = 20,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray, Dict[str, Any]]:
    """Execute K-Means++ initialization clustering."""
    return run_kmeans(
        X=X,
        k=k,
        init="k-means++",
        n_init=n_init,
        random_state=random_state
    )

def evaluate_k_range(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    init: str = "k-means++",
    n_init: int = 20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, int]:
    """
    Evaluate clustering metrics across K in [k_min, k_max].
    Calculates Inertia, Silhouette Score, Davies-Bouldin Index,
    and determines the optimal K using peak Silhouette Score.
    """
    records = []
    prev_inertia = None
    
    for k in range(k_min, k_max + 1):
        model = KMeans(
            n_clusters=k,
            init=init,
            n_init=n_init,
            random_state=random_state
        )
        labels = model.fit_predict(X)
        
        sil = float(silhouette_score(X, labels))
        db = float(davies_bouldin_score(X, labels))
        inertia = float(model.inertia_)
        
        pct_reduction = None
        if prev_inertia is not None and prev_inertia > 0:
            pct_reduction = ((prev_inertia - inertia) / prev_inertia) * 100.0
        prev_inertia = inertia
        
        records.append({
            "k": k,
            "silhouette": sil,
            "inertia": inertia,
            "inertia_reduction_pct": pct_reduction,
            "davies_bouldin": db
        })
        
    df_results = pd.DataFrame(records)
    
    # Optimal K determined by highest Silhouette Score
    optimal_row = df_results.loc[df_results["silhouette"].idxmax()]
    optimal_k = int(optimal_row["k"])
    
    return df_results, optimal_k

def run_dbscan(
    X: np.ndarray,
    eps: float = 1.8,
    min_samples: int = 10
) -> Tuple[DBSCAN, np.ndarray, Dict[str, Any]]:
    """
    Execute DBSCAN clustering with robust edge-case handling.
    Correctly identifies density noise points (label == -1) and calculates
    valid silhouette and Davies-Bouldin scores on clustered samples.
    """
    t0 = time.time()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    elapsed = time.time() - t0
    
    total_samples = len(labels)
    unique_labels = set(labels)
    n_noise = int(np.sum(labels == -1))
    noise_pct = (n_noise / total_samples) * 100.0 if total_samples > 0 else 0.0
    
    cluster_labels_set = [l for l in unique_labels if l != -1]
    n_clusters = len(cluster_labels_set)
    
    # Calculate silhouette and Davies-Bouldin safely
    sil = None
    db_index = None
    valid_mask = labels != -1
    n_clustered = int(np.sum(valid_mask))
    
    if n_clusters >= 2 and n_clustered > n_clusters:
        try:
            sil = float(silhouette_score(X[valid_mask], labels[valid_mask]))
            db_index = float(davies_bouldin_score(X[valid_mask], labels[valid_mask]))
        except Exception:
            sil = None
            db_index = None
            
    metrics = {
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_percentage": noise_pct,
        "silhouette": sil,
        "davies_bouldin": db_index,
        "execution_time_sec": elapsed,
        "cluster_counts": pd.Series(labels).value_counts().sort_index().to_dict()
    }
    return db, labels, metrics
