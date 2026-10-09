# Village Morphology Clustering Summary

Workflow for clustering outputs:

```text
IQR winsorization -> RobustScaler -> PCA_85 -> K-Means
```

Result folders:

- `k2/`
- `k3/`
- `k_selection/`  (K=2..6 轮廓系数与 Davies-Bouldin 对比，用于论证 K=3 的合理性)

Metric summary: `k2_k3_clustering_metrics_summary.xlsx`
K-selection detail: `k_selection/k_selection_metrics.xlsx` + `k_selection/k_selection_explanation.md`
