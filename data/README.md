# Data Directory

This directory stores demonstration and sample biological expression data for the **Clustering & PCA Analysis** application.

## Demonstration Dataset Specification
- **Format**: Comma-Separated Values (`.csv`)
- **Samples**: 1,200 biological tissue specimens
- **Features**: 500 numerical transcript / gene expression features (`GENE_001` to `GENE_500`)
- **Metadata**: `SAMPLE_ID` (unique observation key), `SUBTYPE_ANNOTATION` (latent biological subgroup for validation)
- **Latent Cohorts**: 4 distinct biological sub-cohorts + 38 density-based outlier observations

> **Note**: In unsupervised clustering workflows, metadata labels are strictly excluded from preprocessing and model training to prevent data leakage.
