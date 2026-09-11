"""
Interactive Visualizations Module (Plotly):
- 2D PCA Scatter plot with cluster coloration and sample hover
- 3D PCA Scatter plot with rotation, zoom, hover, and cluster legends
- Silhouette Score vs Cluster Count K plot
- K-Means Inertia / Elbow curve with percentage reduction
- 50-Run Stability box plot and run-by-run sequence
- Cluster size distribution bar chart (with DBSCAN noise distinction)
- Top-variable feature correlation heatmap
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Consistent color palette for up to 10 clusters + noise
CLUSTER_COLORS = [
    "#2563eb",  # Blue (Cluster 0)
    "#10b981",  # Emerald (Cluster 1)
    "#f59e0b",  # Amber (Cluster 2)
    "#8b5cf6",  # Violet (Cluster 3)
    "#ec4899",  # Pink (Cluster 4)
    "#06b6d4",  # Cyan (Cluster 5)
    "#84cc16",  # Lime (Cluster 6)
    "#f97316",  # Orange (Cluster 7)
]
NOISE_COLOR = "#94a3b8"  # Slate Gray for DBSCAN noise (-1)

def plot_pca_2d(
    pca_df: pd.DataFrame,
    color_col: str = "cluster",
    sample_id_col: str = "sample_id",
    var_ratios: Tuple[float, float] = (0.428, 0.241),
    title: str = "2D PCA Projection"
) -> go.Figure:
    """Create interactive 2D PCA scatter plot with sample hover metadata."""
    fig = go.Figure()
    
    unique_clusters = sorted(pca_df[color_col].unique())
    for i, cl in enumerate(unique_clusters):
        subset = pca_df[pca_df[color_col] == cl]
        
        if cl == -1 or cl == "Noise":
            name = "Noise / Outliers"
            color = NOISE_COLOR
        else:
            name = f"Cluster {cl}"
            color = CLUSTER_COLORS[int(cl) % len(CLUSTER_COLORS)] if isinstance(cl, (int, np.integer)) else CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            
        fig.add_trace(go.Scatter(
            x=subset["PC1"],
            y=subset["PC2"],
            mode="markers",
            name=name,
            marker=dict(
                size=7,
                color=color,
                opacity=0.85,
                line=dict(width=0.5, color="#ffffff")
            ),
            text=subset[sample_id_col] if sample_id_col in subset.columns else None,
            hovertemplate=(
                "<b>%{text}</b><br>"
                + f"<b>{name}</b><br>"
                + f"PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"
            )
        ))
        
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, weight="bold")),
        xaxis=dict(
            title=f"PC1 ({var_ratios[0] * 100:.1f}% Variance Explained)",
            gridcolor="rgba(148, 163, 184, 0.15)",
            zerolinecolor="rgba(148, 163, 184, 0.25)"
        ),
        yaxis=dict(
            title=f"PC2 ({var_ratios[1] * 100:.1f}% Variance Explained)",
            gridcolor="rgba(148, 163, 184, 0.15)",
            zerolinecolor="rgba(148, 163, 184, 0.25)"
        ),
        legend=dict(
            title=dict(text="Cohorts"),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(148, 163, 184, 0.3)",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="closest",
        template="plotly_white"
    )
    return fig

def plot_pca_3d(
    pca_df: pd.DataFrame,
    color_col: str = "cluster",
    sample_id_col: str = "sample_id",
    var_ratios: Tuple[float, float, float] = (0.428, 0.241, 0.135),
    title: str = "3D PCA Projection — Cluster Geometry"
) -> go.Figure:
    """Create interactive 3D PCA scatter plot with full rotation, zoom, and inspection."""
    fig = go.Figure()
    
    unique_clusters = sorted(pca_df[color_col].unique())
    for i, cl in enumerate(unique_clusters):
        subset = pca_df[pca_df[color_col] == cl]
        
        if cl == -1 or cl == "Noise":
            name = "Noise / Outliers"
            color = NOISE_COLOR
        else:
            name = f"Cluster {cl}"
            color = CLUSTER_COLORS[int(cl) % len(CLUSTER_COLORS)] if isinstance(cl, (int, np.integer)) else CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            
        fig.add_trace(go.Scatter3d(
            x=subset["PC1"],
            y=subset["PC2"],
            z=subset["PC3"],
            mode="markers",
            name=name,
            marker=dict(
                size=4.5,
                color=color,
                opacity=0.88,
                line=dict(width=0.5, color="#ffffff")
            ),
            text=subset[sample_id_col] if sample_id_col in subset.columns else None,
            hovertemplate=(
                "<b>%{text}</b><br>"
                + f"<b>{name}</b><br>"
                + "PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>"
            )
        ))
        
    cum_var = sum(var_ratios) * 100.0
    fig.update_layout(
        title=dict(
            text=f"{title} (Cumulative Variance: {cum_var:.1f}%)",
            font=dict(size=16, weight="bold")
        ),
        scene=dict(
            xaxis=dict(title=f"PC1 ({var_ratios[0] * 100:.1f}%)"),
            yaxis=dict(title=f"PC2 ({var_ratios[1] * 100:.1f}%)"),
            zaxis=dict(title=f"PC3 ({var_ratios[2] * 100:.1f}%)"),
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2)
            )
        ),
        legend=dict(
            title=dict(text="Cohorts"),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(148, 163, 184, 0.3)",
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white"
    )
    return fig

def plot_silhouette_curve(
    k_df: pd.DataFrame,
    optimal_k: int
) -> go.Figure:
    """Create interactive Silhouette Score vs Cluster Count line chart with optimal marker."""
    fig = go.Figure()
    
    # Line chart of all scores
    fig.add_trace(go.Scatter(
        x=k_df["k"],
        y=k_df["silhouette"],
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=9, color="#1d4ed8"),
        hovertemplate="k=%{x}<br>Silhouette: %{y:.3f}<extra></extra>"
    ))
    
    # Highlight optimal K
    opt_row = k_df[k_df["k"] == optimal_k].iloc[0]
    fig.add_trace(go.Scatter(
        x=[optimal_k],
        y=[opt_row["silhouette"]],
        mode="markers",
        name=f"Optimal K = {optimal_k}",
        marker=dict(
            size=15,
            color="#ef4444",
            symbol="star",
            line=dict(width=2, color="#ffffff")
        ),
        hovertemplate=f"<b>Optimal K = {optimal_k}</b><br>Peak Silhouette: {opt_row['silhouette']:.3f}<extra></extra>"
    ))
    
    fig.add_annotation(
        x=optimal_k,
        y=opt_row["silhouette"],
        text=f"Optimal K={optimal_k} (Sil={opt_row['silhouette']:.3f})",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-35,
        font=dict(size=12, color="#ef4444", weight="bold"),
        bgcolor="rgba(254, 242, 242, 0.9)",
        bordercolor="#fca5a5"
    )
    
    fig.update_layout(
        title=dict(text="Silhouette Coefficient vs Cluster Count (K)", font=dict(size=16, weight="bold")),
        xaxis=dict(title="Number of Clusters (K)", dtick=1),
        yaxis=dict(title="Silhouette Coefficient", range=[0.0, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white"
    )
    return fig

def plot_elbow_curve(k_df: pd.DataFrame) -> go.Figure:
    """Create interactive Elbow (Inertia vs K) plot with percentage reduction."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=k_df["k"],
        y=k_df["inertia"],
        mode="lines+markers",
        name="Inertia (WCSS)",
        line=dict(color="#10b981", width=3),
        marker=dict(size=9, color="#047857"),
        hovertemplate="k=%{x}<br>Inertia: %{y:.1f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text="K-Means Inertia (Elbow Curve) vs Cluster Count (K)", font=dict(size=16, weight="bold")),
        xaxis=dict(title="Number of Clusters (K)", dtick=1),
        yaxis=dict(title="Inertia (Within-Cluster Sum of Squares)"),
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white"
    )
    return fig

def plot_stability_distribution(df_runs: pd.DataFrame) -> go.Figure:
    """Create box plot with overlaid jitter points for 50-run silhouette stability."""
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=df_runs["silhouette"],
        name="50-Run Silhouette Distribution",
        boxpoints="all",
        jitter=0.4,
        pointpos=-1.8,
        marker=dict(color="#6366f1", size=6, opacity=0.7),
        line=dict(color="#4338ca", width=2),
        fillcolor="rgba(99, 102, 241, 0.25)"
    ))
    
    mean_sil = df_runs["silhouette"].mean()
    fig.add_hline(
        y=mean_sil,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Mean: {mean_sil:.4f}",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title=dict(text="Clustering Stability: Silhouette Distribution Across 50 Seeds", font=dict(size=15, weight="bold")),
        yaxis=dict(title="Silhouette Score"),
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white"
    )
    return fig

def plot_stability_run_sequence(df_runs: pd.DataFrame) -> go.Figure:
    """Create run-by-run silhouette score line chart across 50 Monte Carlo seeds."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_runs["run"],
        y=df_runs["silhouette"],
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color="#8b5cf6", width=2),
        marker=dict(size=6, color="#6d28d9"),
        hovertemplate="Run #%{x}<br>Silhouette: %{y:.4f}<extra></extra>"
    ))
    
    mean_sil = df_runs["silhouette"].mean()
    fig.add_hline(
        y=mean_sil,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Mean = {mean_sil:.4f}",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title=dict(text="Run-by-Run Stability (50 Independent Random Initializations)", font=dict(size=15, weight="bold")),
        xaxis=dict(title="Monte Carlo Run Index (1 to 50)", dtick=5),
        yaxis=dict(title="Silhouette Score"),
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white"
    )
    return fig

def plot_cluster_sizes_bar(
    counts_dict: Dict[Any, int],
    title: str = "Cluster Size Distribution",
    is_dbscan: bool = False
) -> go.Figure:
    """Create bar chart of cluster sample counts with distinct coloring for noise."""
    clusters = list(counts_dict.keys())
    counts = list(counts_dict.values())
    total = sum(counts)
    
    colors = []
    labels = []
    for c in clusters:
        if c == -1 or c == "Noise":
            labels.append("Noise / Outlier (-1)")
            colors.append(NOISE_COLOR)
        else:
            labels.append(f"Cluster {c}")
            idx = int(c) if str(c).isdigit() else 0
            colors.append(CLUSTER_COLORS[idx % len(CLUSTER_COLORS)])
            
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=counts,
            text=[f"{cnt} ({cnt/total*100:.1f}%)" for cnt in counts],
            textposition="auto",
            marker=dict(color=colors, line=dict(width=1, color="#ffffff")),
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, weight="bold")),
        xaxis=dict(title="Sub-Cohort"),
        yaxis=dict(title="Number of Observations"),
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white"
    )
    return fig

def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: List[str],
    top_n: int = 25
) -> go.Figure:
    """
    Subsampled correlation heatmap for top N variable features
    to maintain fast application performance.
    """
    # Select top N most variable features
    variances = df[feature_cols].var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    
    corr_matrix = df[top_features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=top_features,
        y=top_features,
        colorscale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(title="Pearson r")
    ))
    
    fig.update_layout(
        title=dict(text=f"Feature Correlation Matrix (Top {top_n} Variable Genes)", font=dict(size=15, weight="bold")),
        xaxis=dict(tickangle=-45),
        margin=dict(l=60, r=40, t=50, b=60),
        template="plotly_white"
    )
    return fig
