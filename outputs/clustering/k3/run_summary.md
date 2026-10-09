# Fixed Village Morphology Clustering Summary

Input file: `data\仅因变量标准化数据.xlsx`

Fixed workflow:

```text
IQR winsorization -> RobustScaler -> PCA_85 -> K-Means -> K=3
```

Metrics:

- PCA components: 4
- PCA explained variance: 0.8809
- Silhouette: 0.2588
- Calinski-Harabasz: 281.2097
- Davies-Bouldin: 1.2607
- Cluster counts: 1:325; 2:266; 3:141

Primary outputs:

- `diagnostics/morphology_diagnostics_iqr_winsor.xlsx`
- `tables/fixed_clustering_metrics.xlsx`
- `tables/village_year_cluster_assignments.xlsx`
- `tables/cluster_profiles.xlsx`
- `tables/pca_explained_variance.xlsx`
- `tables/pca_loadings.xlsx`
- `figures/iqr_winsorization_effect.png`
- `figures/pca_explained_variance.png`
- `figures/pca_cluster_scatter.png`
- `figures/cluster_center_radar.png`
- `figures/cluster_transition_heatmaps.png`
