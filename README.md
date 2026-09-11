# Clustering & PCA Analysis — High-Dimensional Biological Data

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-ready **Streamlit web application** for unsupervised patient sub-cohort discovery and high-dimensional transcriptomic pattern recognition.

---

## 1. Project Overview

High-dimensional biological and genetic expression profiles suffer from the **curse of dimensionality**, high multicollinearity, and experimental noise. Manual categorization of biological tissue samples into disease or response subtypes is intractable when hundreds or thousands of transcript-level features are present.

This application provides a comprehensive analytical pipeline to discover latent patient/tissue sub-cohorts using unsupervised learning while providing quantitative evidence that discovered clusters are statistically robust, stable, and well-separated.

---

## 2. Business & Research Problem

* **Dimensionality Challenge**: 500+ gene expression features across biological specimens.
* **Objective**: Discover latent disease sub-cohorts without relying on supervised labels (zero data leakage).
* **Validation**: Quantify cluster separation via Silhouette Analysis and Davies-Bouldin metrics.
* **Stability Verification**: Perform a 50-run Monte Carlo stability test across varied initializations to ensure discoveries are not random artifacts.
* **Outlier Discrimination**: Isolate atypical/degraded tissue specimens using density-based DBSCAN.

---

## 3. Key Analytical Features

- **Dynamic Data Ingestion**: Supports custom CSV uploads or an automated realistic synthetic biological demonstration dataset (1,200 samples × 500 features).
- **Leakage-Free Preprocessing**: Automated numerical feature detection, missing value imputation (Mean/Median), zero-variance pruning, optional `log1p` transformation, and `StandardScaler`.
- **Quantitative K Optimization**: Sweeps $K \in [2, 8]$ using K-Means++ and automatically identifies the optimal cluster count using peak Silhouette Score.
- **Dimensionality Reduction**: 3-component PCA capturing dominant biological covariance (~80.4% variance explained).
- **Interactive Visualizations**: 2D and 3D interactive Plotly projections with camera rotation, zoom, hover tooltips, and cluster legends.
- **50-Run Stability Benchmark**: True Monte Carlo simulation running K-Means++ across 50 genuine random seeds with distribution box plots and run sequences.
- **DBSCAN Outlier Detection**: Density-based identification of atypical observations.
- **Multi-Format Export Center**: Download clustered data, model metrics, PCA coordinates, and stability results as CSV files.

---

## 4. Benchmark Demonstration Targets

| Metric | Target Reference | Description |
| :--- | :--- | :--- |
| **Samples** | `1,200` | Tissue observations |
| **Features** | `500` | Gene expression variables |
| **Optimal K** | `4` | Latent biological sub-cohorts |
| **Peak Silhouette** | `~0.690` | Cluster separation quality |
| **K-Means++ Inertia** | `~640` | Within-cluster sum of squares |
| **3D PCA Variance** | `~80.4%` | PC1 ~42.8%, PC2 ~24.1%, PC3 ~13.5% |
| **DBSCAN Noise** | `~38` | Atypical outlier specimens |

---

## 5. Project Architecture

```text
clustering-pca-streamlit/
│
├── app.py                     # Streamlit application entrypoint & navigation
├── requirements.txt           # Pinned compatible dependencies
├── README.md                  # Project documentation & deployment guide
├── .gitignore                 # Version control exclusions
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── utils.py               # Seed management, styling, synthetic data generator
│   ├── preprocessing.py       # Imputation, zero-var filter, log1p, standardization
│   ├── clustering.py          # K-Means, K-Means++, DBSCAN execution engines
│   ├── pca_analysis.py        # PCA decomposition, explained variance, loadings
│   ├── evaluation.py          # Silhouette, Davies-Bouldin, 50-run stability testing
│   └── visualization.py       # Plotly interactive 2D/3D charts and diagnostic plots
│
├── data/
│   └── README.md              # Dataset guidance & structure
│
└── assets/
    └── README.md              # Static media & architectural assets
```

---

## 6. Installation & Local Setup

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Step-by-Step Local Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rupesh4113/Clustering-PCA-Analysis.git
   cd Clustering-PCA-Analysis
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`.

---

## 7. Streamlit Community Cloud Deployment

Deploying this application to **Streamlit Community Cloud** takes less than 3 minutes:

1. **Push to GitHub**:
   Ensure all files (`app.py`, `requirements.txt`, `src/`, `README.md`) are committed and pushed to your GitHub repository.
2. **Access Streamlit Cloud**:
   Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. **Create New App**:
   Click **"New app"** / **"Deploy an app"**.
4. **Select Repository**:
   - Repository: `Rupesh4113/Clustering-PCA-Analysis`
   - Branch: `main` (or `master`)
   - Main file path: `app.py`
5. **Deploy**:
   Click **"Deploy!"**. Streamlit will automatically install `requirements.txt` and launch the live web application.

---

## 8. Methodology & Reproducibility

- **Global Seed**: `RANDOM_STATE = 42` (Configurable via sidebar).
- **Zero Data Leakage**: Target and diagnostic columns are strictly excluded from clustering matrices.
- **Model Evaluation**: Evaluated using Silhouette Coefficient and Davies-Bouldin Index across multiple random initializations.
- **Disclaimer**: Unsupervised clustering discovers mathematical groupings. Discovered cohorts should be verified against clinical outcomes or orthogonal biomarkers.

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
