"""K=3 K-means 随机种子稳定性检验（ARI 对比参考方案）。

背景：论文最终使用的是固定 random_state=42 的 K=3 聚类结果作为"参考方案"。
K-means 对初始质心随机敏感，本脚本用 50 个不同随机种子（0-49）重新运行 K=3，
计算每次结果与参考方案的 Adjusted Rand Index (ARI)，用以说明聚类结构是
真实稳定、不依赖随机初始化的。仅报告 ARI 统计量，不涉及 NMI/Jaccard 等。
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "clustering" / "ari_stability"

MORPHOLOGY_COLUMNS = ["MBC", "LCC", "GCC", "INT", "CSR", "IR", "AEL"]
REQUIRED_COLUMNS = ["ID", "YEAR", *MORPHOLOGY_COLUMNS]
REFERENCE_SEED = 42
PCA_VARIANCE_THRESHOLD = 0.85
N_SEEDS = 50
K = 3


def find_morphology_file(data_dir: Path) -> Path:
    for path in sorted(p for p in data_dir.glob("*.xlsx") if not p.name.startswith("~$")):
        try:
            cols = pd.read_excel(path, nrows=0).columns.astype(str).tolist()
        except Exception:
            continue
        if set(REQUIRED_COLUMNS).issubset(cols):
            return path
    raise FileNotFoundError("No Excel file with required morphology columns found.")


def apply_iqr_winsorization(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in MORPHOLOGY_COLUMNS:
        q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
        iqr = q3 - q1
        out[col] = out[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return out


def reorder_labels_by_original_profile(labels: np.ndarray, scaled: np.ndarray) -> np.ndarray:
    """与 pipeline 一致的标签重排，确保不同种子的簇标签处于同一规范顺序。"""
    profile = pd.DataFrame(scaled, columns=MORPHOLOGY_COLUMNS)
    profile["raw_label"] = labels
    scores = profile.groupby("raw_label")[MORPHOLOGY_COLUMNS].mean().mean(axis=1)
    ordered = scores.sort_values().index.tolist()
    label_map = {raw: i + 1 for i, raw in enumerate(ordered)}
    return np.array([label_map[l] for l in labels])


def main() -> None:
    src = find_morphology_file(DATA_DIR)
    raw = pd.read_excel(src)
    features = raw[MORPHOLOGY_COLUMNS].copy().fillna(raw[MORPHOLOGY_COLUMNS].median())

    transformed = apply_iqr_winsorization(features)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(transformed)
    pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=REFERENCE_SEED)
    pca_features = pca.fit_transform(scaled)

    # 参考方案：论文最终使用的 K=3 (random_state=42)
    ref_model = KMeans(n_clusters=K, random_state=REFERENCE_SEED, n_init=50)
    ref_raw = ref_model.fit_predict(pca_features)
    ref_labels = reorder_labels_by_original_profile(ref_raw, scaled)

    # 50 个随机种子
    seeds = list(range(N_SEEDS))
    ari_values = []
    rows = []
    for seed in seeds:
        model = KMeans(n_clusters=K, random_state=seed, n_init=50)
        raw_labels = model.fit_predict(pca_features)
        labels = reorder_labels_by_original_profile(raw_labels, scaled)
        ari = adjusted_rand_score(ref_labels, labels)
        ari_values.append(ari)
        rows.append({"seed": seed, "ARI_vs_reference": ari,
                     "cluster_counts": "; ".join(
                         f"{int(k)}:{int(v)}" for k, v in
                         pd.Series(labels).value_counts().sort_index().items())})

    arr = np.array(ari_values)
    summary = {
        "Metric": ["Median", "25th Percentile", "75th Percentile", "Min", "Max"],
        "ARI": [
            float(np.median(arr)),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)),
            float(arr.min()),
            float(arr.max()),
        ],
    }
    summary_df = pd.DataFrame(summary)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(OUT_DIR / "ari_per_seed.xlsx", index=False)
    summary_df.to_excel(OUT_DIR / "ari_summary.xlsx", index=False)

    # 文本报告
    report = (
        "# ARI 稳定性检验结果 (K=3, 50 随机种子 vs 参考方案 seed=42)\n\n"
        f"参考方案：random_state={REFERENCE_SEED}，K={K}，PCA_85 降维空间\n"
        f"评估种子数：{N_SEEDS} (seed 0-49)\n\n"
        "## 统计量\n\n"
        "| 统计量 | ARI |\n|---|---|\n"
        f"| Median | {summary_df.loc[0, 'ARI']:.4f} |\n"
        f"| 25th Percentile | {summary_df.loc[1, 'ARI']:.4f} |\n"
        f"| 75th Percentile | {summary_df.loc[2, 'ARI']:.4f} |\n"
        f"| Min | {summary_df.loc[3, 'ARI']:.4f} |\n"
        f"| Max | {summary_df.loc[4, 'ARI']:.4f} |\n\n"
        "结论：ARI 越接近 1 表示与参考方案越一致；若中位数接近 1 且波动很小，\n"
        "说明 K=3 聚类结果稳定、不依赖随机初始化。\n"
    )
    (OUT_DIR / "ari_stability_report.md").write_text(report, encoding="utf-8")

    print("ARI stability test completed. Outputs in:", OUT_DIR)
    print(report)


if __name__ == "__main__":
    main()
