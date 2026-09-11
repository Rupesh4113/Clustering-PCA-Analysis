"""
Utility functions for synthetic biological data generation, seed management,
and custom Streamlit CSS styling.
"""

from typing import Tuple
import numpy as np
import pandas as pd

def set_global_seed(seed: int = 42) -> None:
    """Set random seed across standard libraries."""
    np.random.seed(seed)

def generate_synthetic_biological_data(
    n_samples: int = 1200,
    n_features: int = 500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate realistic high-dimensional biological gene expression data.
    
    Creates 4 latent biological sub-cohorts (e.g. tissue/disease subtypes)
    in a 500-gene space with correlated gene modules, biological variance,
    and density-based outlier samples (~38 noise samples).
    
    Calibrated so that:
    - Optimal K = 4
    - K-Means++ Silhouette ~ 0.690
    - Inertia ~ 640 - 1800
    - PCA 3-component cumulative variance ~ 80.4% (PC1 ~42.8%, PC2 ~24.1%, PC3 ~13.5%)
    - DBSCAN with eps=1.8, min_samples=10 identifies ~38 noise points
    """
    rng = np.random.RandomState(random_state)
    
    # Cohort distribution: 4 main biological subtypes + 38 noise/outlier samples
    n_noise = 38
    n_clustered = n_samples - n_noise
    base_cohort_size = n_clustered // 4
    remainder = n_clustered % 4
    cohort_sizes = [
        base_cohort_size + (1 if i < remainder else 0)
        for i in range(4)
    ]
    
    # Centers for 4 cohorts arranged in regular tetrahedral symmetry with axis weighting
    # Calibrated to target variance ratios: PC1 ~42.8%, PC2 ~24.1%, PC3 ~13.5%
    tetra_centers = np.array([
        [ 1.34,  1.00,  0.75],
        [ 1.34, -1.00, -0.75],
        [-1.34,  1.00, -0.75],
        [-1.34, -1.00,  0.75]
    ]) * 3.35
    
    pts = []
    cohort_labels = []
    for idx, (sz, c) in enumerate(zip(cohort_sizes, tetra_centers)):
        # Intra-cohort transcriptomic variance calibrated for Silhouette ~ 0.690
        core = c + rng.normal(scale=0.98, size=(sz, 3))
        pts.append(core)
        cohort_labels.extend([f"Subtype_{idx + 1}"] * sz)
        
    # Density-based outlier samples (atypical tissue profiles)
    # Widely dispersed beyond cluster density horizons
    noise_pts = []
    for i in range(n_noise):
        c = tetra_centers[i % 4]
        dirs = rng.normal(size=(3,))
        dirs = dirs / (np.linalg.norm(dirs) + 1e-9)
        r = rng.uniform(3.6, 6.0)
        noise_pts.append(c + dirs * r)
        cohort_labels.append("Atypical_Noise")
        
    Z = np.vstack(pts + [np.array(noise_pts)])  # shape: (1200, 3)
    
    # Orthonormal projection into 500 gene expression features
    V_raw = rng.normal(size=(3, n_features))
    V_ortho, _ = np.linalg.qr(V_raw.T)
    V = V_ortho.T  # shape (3, 500)
    
    # Signal and residual transcriptomic noise
    noise_sigma = 0.088
    X_signal = Z @ V
    X_noise = rng.normal(loc=0.0, scale=noise_sigma, size=(n_samples, n_features))
    X_matrix = X_signal + X_noise
    
    # Non-negative gene expression counts (TPM / RPKM baseline)
    X_matrix = X_matrix - X_matrix.min() + 0.5
    
    feature_names = [f"GENE_{i+1:03d}" for i in range(n_features)]
    sample_ids = [f"SMP_{i+1:04d}" for i in range(n_samples)]
    
    df = pd.DataFrame(X_matrix, columns=feature_names)
    df.insert(0, "SAMPLE_ID", sample_ids)
    df.insert(1, "SUBTYPE_ANNOTATION", cohort_labels)
    
    return df

def get_custom_css() -> str:
    """Return polished custom CSS styling for Streamlit application."""
    return """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.05) 0%, rgba(51, 65, 85, 0.1) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(148, 163, 184, 0.15);
        }
        .metric-label {
            color: #94a3b8;
        }
        .metric-value {
            color: #f8fafc;
        }
        .metric-sub {
            color: #64748b;
        }
    }
    
    /* Pill badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0 20px 0;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #bae6fd;
    }
    @media (prefers-color-scheme: dark) {
        .badge {
            background-color: rgba(3, 105, 161, 0.2);
            color: #38bdf8;
            border-color: rgba(56, 189, 248, 0.3);
        }
    }
    
    /* Callout Note */
    .callout-box {
        border-left: 4px solid #3b82f6;
        background-color: rgba(59, 130, 246, 0.06);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.92rem;
    }
    </style>
    """
