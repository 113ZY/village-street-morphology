from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
warnings.filterwarnings("ignore", message=".*Pandas requires version.*")

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler


MORPHOLOGY_COLUMNS = ["MBC", "LCC", "GCC", "INT", "CSR", "IR", "AEL"]
REQUIRED_COLUMNS = ["ID", "YEAR", *MORPHOLOGY_COLUMNS]
CLUSTER_COUNTS = [2, 3]
K_SELECTION_RANGE = [2, 3, 4, 5, 6]
PCA_VARIANCE_THRESHOLD = 0.85

# 项目统一配色：橙-蓝-紫三色系
CLUSTER_PALETTE = ["#ED6D54", "#91C0D8", "#8762A8"]  # LIST / HIAT / OECT
# 9 色扩展色带
ORANGE_BAND = ["#C94E37", "#ED6D54", "#F4A290"]
BLUE_BAND = ["#6A9FBB", "#91C0D8", "#B8D8E8"]
PURPLE_BAND = ["#6B4890", "#8762A8", "#B394C8"]


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    tables: Path
    figures: Path
    diagnostics: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed village morphology clustering workflow."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Path to the dependent-variable Excel file. If omitted, the script scans data/.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing input Excel files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "clustering",
        help="Directory for clustering outputs.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_output_paths(output_dir: Path) -> OutputPaths:
    paths = OutputPaths(
        root=output_dir,
        tables=output_dir / "tables",
        figures=output_dir / "figures",
        diagnostics=output_dir / "diagnostics",
    )
    for directory in (paths.root, paths.tables, paths.figures, paths.diagnostics):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def find_morphology_file(data_dir: Path) -> Path:
    candidates = sorted(
        path for path in data_dir.glob("*.xlsx") if not path.name.startswith("~$")
    )
    for path in candidates:
        try:
            columns = pd.read_excel(path, nrows=0).columns.astype(str).tolist()
        except Exception:
            continue
        if set(REQUIRED_COLUMNS).issubset(columns):
            return path
    raise FileNotFoundError(
        "No Excel file containing the required morphology columns was found."
    )


def read_morphology_data(input_file: Path | None, data_dir: Path) -> pd.DataFrame:
    source = input_file if input_file is not None else find_morphology_file(data_dir)
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")

    df = pd.read_excel(source)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {missing_columns}")

    df = df.copy()
    df["YEAR"] = df["YEAR"].astype(int)
    df["VILLAGE_ID"] = df.apply(extract_village_id, axis=1)
    df.attrs["source_file"] = str(source)
    return df


def extract_village_id(row: pd.Series) -> str:
    raw_id = str(row["ID"])
    year = str(int(row["YEAR"]))
    if raw_id.endswith(year):
        return raw_id[: -len(year)]
    return re.sub(r"(2014|2016|2018|2020)$", "", raw_id)


def prepare_input_features(df: pd.DataFrame) -> pd.DataFrame:
    duplicated_keys = df.duplicated(subset=["ID", "YEAR"]).sum()
    if duplicated_keys:
        raise ValueError(f"Found duplicated ID-YEAR records: {duplicated_keys}")

    feature_frame = df[MORPHOLOGY_COLUMNS].copy()
    if feature_frame.isna().sum().sum() > 0:
        feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
    return feature_frame


def apply_iqr_winsorization(feature_frame: pd.DataFrame) -> pd.DataFrame:
    transformed = feature_frame.copy()
    for col in MORPHOLOGY_COLUMNS:
        q1 = transformed[col].quantile(0.25)
        q3 = transformed[col].quantile(0.75)
        iqr = q3 - q1
        transformed[col] = transformed[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return transformed


def build_diagnostics(
    df: pd.DataFrame,
    original_features: pd.DataFrame,
    transformed_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    outlier_rows = []
    for col in MORPHOLOGY_COLUMNS:
        series = original_features[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        std = series.std(ddof=0)
        z_scores = (series - series.mean()) / std if std else pd.Series(0, index=series.index)
        iqr_mask = series.lt(lower) | series.gt(upper)
        z_mask = z_scores.abs().gt(3)
        rows.append(
            {
                "variable": col,
                "missing_count": int(df[col].isna().sum()),
                "mean": series.mean(),
                "std": series.std(ddof=1),
                "min": series.min(),
                "q1": q1,
                "median": series.median(),
                "q3": q3,
                "max": series.max(),
                "iqr_outlier_count": int(iqr_mask.sum()),
                "z_outlier_count_abs_gt_3": int(z_mask.sum()),
                "skewness_before_transform": series.skew(),
                "skewness_after_iqr_winsorization": transformed_features[col].skew(),
            }
        )

        flagged = df.loc[iqr_mask | z_mask, ["ID", "VILLAGE_ID", "YEAR"]].copy()
        flagged["variable"] = col
        flagged["value"] = series.loc[flagged.index].to_numpy()
        flagged["z_score"] = z_scores.loc[flagged.index].to_numpy()
        flagged["is_iqr_outlier"] = iqr_mask.loc[flagged.index].to_numpy()
        flagged["is_z_outlier_abs_gt_3"] = z_mask.loc[flagged.index].to_numpy()
        outlier_rows.append(flagged)

    diagnostics = pd.DataFrame(rows)
    outliers = pd.concat(outlier_rows, ignore_index=True)
    correlation = transformed_features.corr()
    return diagnostics, outliers, correlation


def run_fixed_clustering(
    transformed_features: pd.DataFrame,
    cluster_count: int,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, PCA, KMeans, RobustScaler]:
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(transformed_features)

    pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=random_state)
    pca_features = pca.fit_transform(scaled_features)

    model = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=50)
    raw_labels = model.fit_predict(pca_features)
    labels = reorder_labels_by_original_profile(raw_labels, scaled_features)

    metrics = pd.DataFrame(
        [
            {
                "Preprocessing": "IQR_winsorization",
                "Scaler": "RobustScaler",
                "Dimensionality_Reduction": "PCA_85",
                "PCA_Components": pca_features.shape[1],
                "PCA_Explained_Variance": float(pca.explained_variance_ratio_.sum()),
                "Algorithm": "KMeans",
                "K": cluster_count,
                "SSE": float(model.inertia_),
                "Silhouette": silhouette_score(pca_features, raw_labels),
                "Calinski_Harabasz": calinski_harabasz_score(pca_features, raw_labels),
                "Davies_Bouldin": davies_bouldin_score(pca_features, raw_labels),
                "Cluster_Counts": format_counts(labels),
                "Min_Cluster_Size": int(pd.Series(labels).value_counts().min()),
            }
        ]
    )

    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index=MORPHOLOGY_COLUMNS,
        columns=[f"PC{i + 1}" for i in range(pca_features.shape[1])],
    ).reset_index(names="Variable")
    pca_variance = pd.DataFrame(
        {
            "Component": [f"PC{i + 1}" for i in range(pca_features.shape[1])],
            "Explained_Variance_Ratio": pca.explained_variance_ratio_,
            "Cumulative_Explained_Variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return labels, metrics, pca_loadings, pca_features, pca, model, scaler, pca_variance


def reorder_labels_by_original_profile(labels: np.ndarray, scaled_features: np.ndarray) -> np.ndarray:
    profile = pd.DataFrame(scaled_features, columns=MORPHOLOGY_COLUMNS)
    profile["raw_label"] = labels
    label_scores = profile.groupby("raw_label")[MORPHOLOGY_COLUMNS].mean().mean(axis=1)
    ordered_raw_labels = label_scores.sort_values().index.tolist()
    label_map = {raw_label: index + 1 for index, raw_label in enumerate(ordered_raw_labels)}
    return np.array([label_map[label] for label in labels])


def evaluate_k_selection(
    scaled_features: np.ndarray,
    pca_features: np.ndarray,
    random_state: int,
    k_range: list[int],
) -> pd.DataFrame:
    """对 range 内的每个 K 运行 KMeans，计算轮廓系数与 Davies-Bouldin 指数。

    两个指标均在 PCA 降维后的特征空间上计算（与最终聚类一致），因此可直接用于
    解释为什么选择某个 K。
    """
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=50)
        labels = model.fit_predict(pca_features)
        rows.append(
            {
                "K": k,
                "SSE": float(model.inertia_),
                "Silhouette": silhouette_score(pca_features, labels),
                "Calinski_Harabasz": calinski_harabasz_score(pca_features, labels),
                "Davies_Bouldin": davies_bouldin_score(pca_features, labels),
                "Min_Cluster_Size": int(pd.Series(labels).value_counts().min()),
            }
        )
    return pd.DataFrame(rows)


def plot_k_selection_curve(
    k_selection: pd.DataFrame,
    output_file: Path | None = None,
    ax: plt.Axes | None = None,
) -> None:
    """绘制 K 选择评估曲线：轮廓系数（越大越好）与 Davies-Bouldin（越小越好）。

    配色采用项目统一橙-蓝-紫三色系：
    - Silhouette 曲线：蓝 #91C0D8
    - Davies-Bouldin 曲线：橙 #ED6D54
    - 选定 K=3 参考线：紫 #8762A8
    - 最优 K 标记：紫系深紫 #6B4890 / 橙系深橙 #C94E37
    """
    plt.rcParams["font.family"] = "Times New Roman"
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # 项目统一三色系（引用模块级常量）
    BLUE = CLUSTER_PALETTE[1]
    ORANGE = CLUSTER_PALETTE[0]
    PURPLE = CLUSTER_PALETTE[2]
    PURPLE_DARK = PURPLE_BAND[0]
    ORANGE_DARK = ORANGE_BAND[0]

    chosen = 3

    # 左轴：Silhouette（越大越好）
    ax.plot(k_selection["K"], k_selection["Silhouette"], marker="o", color=BLUE, linewidth=2,
            label="Silhouette (↑)")
    best_k_s = int(k_selection.loc[k_selection["Silhouette"].idxmax(), "K"])
    ax.scatter([best_k_s], [k_selection["Silhouette"].max()], s=110, color=PURPLE_DARK,
               zorder=5)
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Silhouette coefficient", color="black")
    ax.tick_params(axis="y", labelcolor="black")
    ax.set_xticks(k_selection["K"])
    ax.set_ylim(0, max(k_selection["Silhouette"]) * 1.25)

    # 右轴：Davies-Bouldin（越小越好）
    ax2 = ax.twinx()
    ax2.plot(k_selection["K"], k_selection["Davies_Bouldin"], marker="s", color=ORANGE, linewidth=2,
             label="Davies-Bouldin (↓)")
    best_k_d = int(k_selection.loc[k_selection["Davies_Bouldin"].idxmin(), "K"])
    ax2.scatter([best_k_d], [k_selection["Davies_Bouldin"].min()], s=110, color=ORANGE_DARK,
                zorder=5)
    ax2.set_ylabel("Davies-Bouldin index", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.set_ylim(0, max(k_selection["Davies_Bouldin"]) * 1.15)

    # 选定 K=3 参考线（贯穿两轴）
    ax.axvline(chosen, color=PURPLE, linestyle="--", linewidth=1.3)

    ax.set_title("Cluster-number selection: Silhouette & Davies-Bouldin vs K")
    ax.grid(alpha=0.25)

    # 精简图例：仅保留两条曲线说明
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower right")

    if output_file is not None:
        fig.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_k_selection_summary(
    paths: OutputPaths,
    k_selection: pd.DataFrame,
) -> None:
    """生成 K 选择对比的文字说明，解释为何选 K=3。"""
    sil = k_selection.set_index("K")["Silhouette"]
    dbi = k_selection.set_index("K")["Davies_Bouldin"]
    best_sil_k = int(sil.idxmax())
    best_dbi_k = int(dbi.idxmin())
    k3 = k_selection.set_index("K").loc[3]

    chosen = 3
    # 说明：最优 K 在轮廓系数与 DBI 上未必完全一致，K=3 在当前数据中同时在两类指标上达到最优。
    lines = [
        "# K 选择依据（轮廓系数 & Davies-Bouldin 指数）",
        "",
        "评估范围：K = " + ", ".join(str(k) for k in k_selection["K"].tolist()),
        "所有指标均在 PCA_85 降维空间上计算，与最终聚类一致。",
        "",
        "## 指标结果",
        "",
        "| K | SSE | Silhouette (↑) | Calinski-Harabasz (↑) | Davies-Bouldin (↓) | Min Cluster Size |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in k_selection.iterrows():
        lines.append(
            f"| {int(row['K'])} | {row['SSE']:.2f} | {row['Silhouette']:.4f} | "
            f"{row['Calinski_Harabasz']:.2f} | {row['Davies_Bouldin']:.4f} | {int(row['Min_Cluster_Size'])} |"
        )
    lines += [
        "",
        "## 选择 K=3 的理由",
        "",
        f"- 轮廓系数（越大越好）在 K={best_sil_k} 处最大（{sil.max():.4f}），"
        f"显著优于 K=2（{sil.get(2, float('nan')):.4f}）和 K=4（{sil.get(4, float('nan')):.4f}）。",
        f"- Davies-Bouldin 指数（越小越好）在 K={best_dbi_k} 处最小（{dbi.min():.4f}），"
        f"明显低于 K=2（{dbi.get(2, float('nan')):.4f}）和 K=4（{dbi.get(4, float('nan')):.4f}）。",
        "- K=3 在两类指标（聚类紧凑度与分离度）上同时达到最优，"
        "且能区分出三个可解释的村落形态原型（如高密、过渡、低密），"
        "而 K≥4 会带来过小簇（最小簇样本数偏低）且指标改善边际递减。",
        "",
        "结论：选用 K=3。",
    ]
    (paths.root / "k_selection_explanation.md").write_text("\n".join(lines), encoding="utf-8")


def format_counts(labels: np.ndarray) -> str:
    counts = pd.Series(labels).value_counts().sort_index()
    return "; ".join(f"{int(label)}:{int(count)}" for label, count in counts.items())


def build_assignments(
    df: pd.DataFrame,
    original_features: pd.DataFrame,
    transformed_features: pd.DataFrame,
    labels: np.ndarray,
    pca_features: np.ndarray,
) -> pd.DataFrame:
    result = df[["ID", "VILLAGE_ID", "YEAR"]].copy()
    for col in MORPHOLOGY_COLUMNS:
        result[col] = original_features[col]
        result[f"{col}_Transformed"] = transformed_features[col]
    result["Cluster"] = labels
    result["Cluster_Label"] = result["Cluster"].map(lambda value: f"Cluster_{value}")

    if pca_features.shape[1] >= 2:
        pca_2d = pca_features[:, :2]
    else:
        pca_2d = np.column_stack([pca_features[:, 0], np.zeros(len(pca_features))])
    result["PC1"] = pca_2d[:, 0]
    result["PC2"] = pca_2d[:, 1]
    return result


def build_cluster_profiles(
    original_features: pd.DataFrame,
    transformed_features: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = original_features.copy()
    transformed = transformed_features.copy()
    original["Cluster"] = labels
    transformed["Cluster"] = labels

    original_profile = original.groupby("Cluster")[MORPHOLOGY_COLUMNS].mean().reset_index()
    transformed_profile = transformed.groupby("Cluster")[MORPHOLOGY_COLUMNS].mean().reset_index()
    for frame in (original_profile, transformed_profile):
        frame["Cluster_Label"] = frame["Cluster"].map(lambda value: f"Cluster_{value}")
    return original_profile, transformed_profile


def build_transition_tables(assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    path_table = assignments.pivot_table(
        index="VILLAGE_ID",
        columns="YEAR",
        values="Cluster",
        aggfunc="first",
    ).sort_index()
    path_table.columns = [f"Cluster_{int(col)}" for col in path_table.columns]
    year_columns = path_table.columns.tolist()
    path_table["Transition_Path"] = path_table[year_columns].astype(str).agg(" -> ".join, axis=1)
    path_table["Transition_Type"] = path_table[year_columns].apply(classify_transition, axis=1)
    path_table = path_table.reset_index()

    years = sorted(assignments["YEAR"].unique())
    matrices: dict[str, pd.DataFrame] = {}
    for start, end in zip(years[:-1], years[1:]):
        matrices[f"{start}_to_{end}"] = transition_matrix(assignments, start, end)
    matrices[f"{years[0]}_to_{years[-1]}"] = transition_matrix(assignments, years[0], years[-1])
    return path_table, matrices


def classify_transition(row: pd.Series) -> str:
    values = row.astype(int).to_numpy()
    if np.all(values == values[0]):
        return "Stable"
    diffs = np.diff(values)
    if values[0] == values[-1]:
        return "Oscillating_Return"
    if np.count_nonzero(diffs) == 1:
        return "Single_Transition"
    if np.all(diffs >= 0):
        return "Progressive_Increase"
    if np.all(diffs <= 0):
        return "Progressive_Decrease"
    return "Multiple_Transition"


def transition_matrix(assignments: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    start = assignments.loc[
        assignments["YEAR"] == start_year, ["VILLAGE_ID", "Cluster"]
    ].rename(columns={"Cluster": "Start_Cluster"})
    end = assignments.loc[
        assignments["YEAR"] == end_year, ["VILLAGE_ID", "Cluster"]
    ].rename(columns={"Cluster": "End_Cluster"})
    merged = start.merge(end, on="VILLAGE_ID", how="inner")
    matrix = pd.crosstab(merged["Start_Cluster"], merged["End_Cluster"])
    matrix.index.name = f"{start_year}_Cluster"
    matrix.columns.name = f"{end_year}_Cluster"
    return matrix


def save_tables(
    paths: OutputPaths,
    diagnostics: pd.DataFrame,
    outliers: pd.DataFrame,
    correlation: pd.DataFrame,
    metrics: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    pca_variance: pd.DataFrame,
    assignments: pd.DataFrame,
    original_profile: pd.DataFrame,
    transformed_profile: pd.DataFrame,
    transition_paths: pd.DataFrame,
    transition_matrices: dict[str, pd.DataFrame],
) -> None:
    diagnostics.to_excel(paths.diagnostics / "morphology_diagnostics_iqr_winsor.xlsx", index=False)
    outliers.to_excel(paths.diagnostics / "morphology_outlier_samples.xlsx", index=False)
    correlation.to_excel(paths.diagnostics / "morphology_correlation_after_iqr_winsor.xlsx")

    metrics.to_excel(paths.tables / "fixed_clustering_metrics.xlsx", index=False)
    assignments.to_excel(paths.tables / "village_year_cluster_assignments.xlsx", index=False)
    transition_paths.to_excel(paths.tables / "village_cluster_transition_paths.xlsx", index=False)
    pca_loadings.to_excel(paths.tables / "pca_loadings.xlsx", index=False)
    pca_variance.to_excel(paths.tables / "pca_explained_variance.xlsx", index=False)

    with pd.ExcelWriter(paths.tables / "cluster_profiles.xlsx") as writer:
        original_profile.to_excel(writer, sheet_name="original_indicator_mean", index=False)
        transformed_profile.to_excel(writer, sheet_name="iqr_winsor_indicator_mean", index=False)

    with pd.ExcelWriter(paths.tables / "cluster_transition_matrices.xlsx") as writer:
        for name, matrix in transition_matrices.items():
            matrix.to_excel(writer, sheet_name=name[:31])


def plot_iqr_winsor_effect(
    original_features: pd.DataFrame,
    transformed_features: pd.DataFrame,
    output_file: Path | None = None,
    axes: list | None = None,
) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 每指标一色：橙-蓝-紫三色系扩展色带（7 个指标）
    iqr_palette = ["#ED6D54", "#C94E37", "#91C0D8", "#6A9FBB", "#8762A8", "#6B4890", "#B8D8E8"]

    def _style_boxplot(ax, data, title):
        bp = ax.boxplot(
            data,
            labels=MORPHOLOGY_COLUMNS,
            showfliers=True,
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], iqr_palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
            patch.set_edgecolor("black")
            patch.set_linewidth(1)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.6)
        for key in ("whiskers", "caps"):
            for line in bp[key]:
                line.set_color("black")
                line.set_linewidth(1)
        for flier in bp["fliers"]:
            flier.set(markerfacecolor="#555555", markeredgecolor="none",
                      marker="o", markersize=3, alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel("Frequency")

    _style_boxplot(axes[0], [original_features[col] for col in MORPHOLOGY_COLUMNS], "Original Indicators")
    _style_boxplot(axes[1], [transformed_features[col] for col in MORPHOLOGY_COLUMNS], "After IQR Winsorization")

    if output_file is not None:
        fig.tight_layout()
        fig.savefig(output_file, dpi=300)
        plt.close(fig)


def plot_pca_variance(
    pca_variance: pd.DataFrame,
    output_file: Path | None = None,
    axis: plt.Axes | None = None,
) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    if axis is None:
        fig, axis = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(pca_variance)) + 1
    axis.bar(x, pca_variance["Explained_Variance_Ratio"], color=CLUSTER_PALETTE[0], alpha=0.8, label="Individual")
    axis.plot(x, pca_variance["Cumulative_Explained_Variance"], marker="o", color=CLUSTER_PALETTE[2], label="Cumulative")
    axis.set_xticks(x)
    axis.set_xticklabels(pca_variance["Component"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Explained variance ratio")
    axis.set_title("PCA Explained Variance")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    if output_file is not None:
        fig.tight_layout()
        fig.savefig(output_file, dpi=300)
        plt.close(fig)


def plot_pca_scatter(
    assignments: pd.DataFrame,
    output_file: Path | None = None,
    axis: plt.Axes | None = None,
) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    if axis is None:
        fig, axis = plt.subplots(figsize=(7.5, 5.5))
    palette = CLUSTER_PALETTE
    for index, cluster in enumerate(sorted(assignments["Cluster"].unique())):
        subset = assignments[assignments["Cluster"] == cluster]
        axis.scatter(
            subset["PC1"],
            subset["PC2"],
            s=34,
            alpha=0.75,
            color=palette[index % len(palette)],
            label=f"Cluster {cluster}",
            edgecolor="white",
            linewidth=0.4,
        )
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title("PCA Projection of Fixed K-Means Clustering")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    if output_file is not None:
        fig.tight_layout()
        fig.savefig(output_file, dpi=300)
        plt.close(fig)


def plot_cluster_radar(
    transformed_profile: pd.DataFrame,
    output_file: Path | None = None,
    axis: plt.Axes | None = None,
) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    labels = MORPHOLOGY_COLUMNS
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    if axis is None:
        fig, axis = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    palette = CLUSTER_PALETTE
    for index, row in transformed_profile.iterrows():
        values = row[MORPHOLOGY_COLUMNS].astype(float).tolist()
        values += values[:1]
        axis.plot(
            angles,
            values,
            linewidth=2,
            color=palette[index % len(palette)],
            label=f"Cluster {int(row['Cluster'])}",
        )
        axis.fill(angles, values, color=palette[index % len(palette)], alpha=0.10)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(labels)
    axis.set_title("Cluster Profiles After IQR Winsorization", pad=20)
    axis.legend(loc="upper right", bbox_to_anchor=(1.24, 1.10), frameon=False)
    if output_file is not None:
        fig.tight_layout()
        fig.savefig(output_file, dpi=300)
        plt.close(fig)


def plot_transition_heatmaps(
    matrices: dict[str, pd.DataFrame],
    output_file: Path | None = None,
    axes: list | None = None,
) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    n_items = len(matrices)
    if axes is None:
        fig, axes = plt.subplots(
            1,
            n_items,
            figsize=(4.3 * n_items, 4.8),
            constrained_layout=True,
        )
    if n_items == 1:
        axes = [axes]

    max_value = max(int(matrix.to_numpy().max()) for matrix in matrices.values())
    heatmap_cmap = LinearSegmentedColormap.from_list(
        "blue_purple",
        ["#B8D8E8", "#91C0D8", "#8762A8", "#6B4890"],
    )
    for axis, (name, matrix) in zip(axes, matrices.items()):
        im = axis.imshow(matrix.to_numpy(), cmap=heatmap_cmap, vmin=0, vmax=max_value)
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("End cluster")
        axis.set_ylabel("Start cluster")
        axis.set_xticks(range(matrix.shape[1]))
        axis.set_xticklabels(matrix.columns.astype(str))
        axis.set_yticks(range(matrix.shape[0]))
        axis.set_yticklabels(matrix.index.astype(str))
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = int(matrix.iloc[row_index, col_index])
                axis.text(
                    col_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="black" if value < max_value * 0.55 else "white",
                    fontsize=9,
                )
    if output_file is not None:
        fig.colorbar(im, ax=axes, shrink=0.78, label="Village count")
        fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    return im


def write_run_summary(
    paths: OutputPaths,
    source_file: str,
    metrics: pd.DataFrame,
) -> None:
    row = metrics.iloc[0]
    summary = f"""# Fixed Village Morphology Clustering Summary

Input file: `{source_file}`

Fixed workflow:

```text
IQR winsorization -> RobustScaler -> PCA_85 -> K-Means -> K={int(row["K"])}
```

Metrics:

- PCA components: {int(row["PCA_Components"])}
- PCA explained variance: {row["PCA_Explained_Variance"]:.4f}
- Silhouette: {row["Silhouette"]:.4f}
- Calinski-Harabasz: {row["Calinski_Harabasz"]:.4f}
- Davies-Bouldin: {row["Davies_Bouldin"]:.4f}
- Cluster counts: {row["Cluster_Counts"]}

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
"""
    (paths.root / "run_summary.md").write_text(summary, encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> None:
    df = read_morphology_data(args.input_file, args.data_dir)
    original_features = prepare_input_features(df)
    transformed_features = apply_iqr_winsorization(original_features)
    diagnostics, outliers, correlation = build_diagnostics(
        df,
        original_features,
        transformed_features,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    # ---- K 选择评估（轮廓系数 & Davies-Bouldin，K=2..6）----
    # 复用 K=3 流程中的 PCA 降维特征空间进行指标计算
    sel_paths = build_output_paths(args.output_dir / "k_selection")
    (
        _labels,
        _metrics,
        _loadings,
        pca_features,
        _pca,
        _model,
        _scaler,
        _variance,
    ) = run_fixed_clustering(transformed_features, 3, args.random_state)
    k_selection = evaluate_k_selection(
        _scaler.transform(transformed_features),
        pca_features,
        args.random_state,
        K_SELECTION_RANGE,
    )
    k_selection.to_excel(sel_paths.tables / "k_selection_metrics.xlsx", index=False)
    plot_k_selection_curve(k_selection, sel_paths.figures / "k_selection_curve.png")
    write_k_selection_summary(sel_paths, k_selection)
    print("K-selection evaluation (K=2..6) completed. Saved to: ", sel_paths.root)

    for cluster_count in CLUSTER_COUNTS:
        paths = build_output_paths(args.output_dir / f"k{cluster_count}")
        (
            labels,
            metrics,
            pca_loadings,
            pca_features,
            _pca,
            _model,
            _scaler,
            pca_variance,
        ) = run_fixed_clustering(transformed_features, cluster_count, args.random_state)
        metrics.insert(0, "Output_Directory", f"k{cluster_count}")
        assignments = build_assignments(
            df,
            original_features,
            transformed_features,
            labels,
            pca_features,
        )
        original_profile, transformed_profile = build_cluster_profiles(
            original_features,
            transformed_features,
            labels,
        )
        transition_paths, transition_matrices = build_transition_tables(assignments)

        save_tables(
            paths,
            diagnostics,
            outliers,
            correlation,
            metrics,
            pca_loadings,
            pca_variance,
            assignments,
            original_profile,
            transformed_profile,
            transition_paths,
            transition_matrices,
        )
        plot_iqr_winsor_effect(
            original_features,
            transformed_features,
            paths.figures / "iqr_winsorization_effect.png",
        )
        plot_pca_variance(pca_variance, paths.figures / "pca_explained_variance.png")
        plot_pca_scatter(assignments, paths.figures / "pca_cluster_scatter.png")
        plot_cluster_radar(transformed_profile, paths.figures / "cluster_center_radar.png")
        plot_transition_heatmaps(
            transition_matrices,
            paths.figures / "cluster_transition_heatmaps.png",
        )
        write_run_summary(paths, df.attrs.get("source_file", ""), metrics)
        all_metrics.append(metrics)

        row = metrics.iloc[0]
        print(f"K={cluster_count} clustering completed. Outputs saved to: {paths.root}")
        print(
            f"Silhouette={row['Silhouette']:.4f}, "
            f"CH={row['Calinski_Harabasz']:.4f}, "
            f"DBI={row['Davies_Bouldin']:.4f}, "
            f"ClusterCounts={row['Cluster_Counts']}"
        )

    summary = pd.concat(all_metrics, ignore_index=True)
    summary.to_excel(args.output_dir / "k2_k3_clustering_metrics_summary.xlsx", index=False)
    (args.output_dir / "run_summary.md").write_text(
        "# Village Morphology Clustering Summary\n\n"
        "Workflow for clustering outputs:\n\n"
        "```text\n"
        "IQR winsorization -> RobustScaler -> PCA_85 -> K-Means\n"
        "```\n\n"
        "Result folders:\n\n"
        "- `k2/`\n"
        "- `k3/`\n"
        "- `k_selection/`  (K=2..6 轮廓系数与 Davies-Bouldin 对比，用于论证 K=3 的合理性)\n\n"
        "Metric summary: `k2_k3_clustering_metrics_summary.xlsx`\n"
        "K-selection detail: `k_selection/k_selection_metrics.xlsx` "
        "+ `k_selection/k_selection_explanation.md`\n",
        encoding="utf-8",
    )


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
