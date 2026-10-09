"""将 PCA 散点、聚类雷达、转移热力图拼成一张长方形组图。

布局：
    +---------------+---------------+
    | PCA scatter   | Cluster radar |   (上：左 / 右)
    +---------------+---------------+
    | Transition heatmaps (跨整行)   |   (下)
    +---------------+---------------+

所有子图统一字体（Times New Roman）、统一刻度字号，边距整齐对齐。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from village_morphology_clustering import pipeline as cl  # noqa: E402

# 统一字体与字号
FONT_FAMILY = "Times New Roman"
TITLE_SIZE = 13
LABEL_SIZE = 12
TICK_SIZE = 11
HEAT_FONT = 11


def _apply_font(ax):
    if ax.title.get_text():
        ax.title.set_fontfamily(FONT_FAMILY)
        ax.title.set_fontsize(TITLE_SIZE)
    if ax.xaxis.get_label().get_text():
        ax.xaxis.label.set_fontfamily(FONT_FAMILY)
        ax.xaxis.label.set_fontsize(LABEL_SIZE)
    if ax.yaxis.get_label().get_text():
        ax.yaxis.label.set_fontfamily(FONT_FAMILY)
        ax.yaxis.label.set_fontsize(LABEL_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)
        label.set_fontsize(TICK_SIZE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cluster-profiles composite figure.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/clustering"),
    )
    args = parser.parse_args()

    # ---- 复用 pipeline 前处理与 K=3 聚类链路 ----
    df = cl.read_morphology_data(None, cl.Path("data"))
    original_features = cl.prepare_input_features(df)
    transformed_features = cl.apply_iqr_winsorization(original_features)

    labels, _metrics, _loadings, pca_features, _pca, _model, _scaler, _pca_variance = (
        cl.run_fixed_clustering(transformed_features, 3, args.random_state)
    )
    assignments = cl.build_assignments(
        df, original_features, transformed_features, labels, pca_features
    )
    _original_profile, transformed_profile = cl.build_cluster_profiles(
        original_features, transformed_features, labels
    )
    _transition_paths, transition_matrices = cl.build_transition_tables(assignments)

    # ---- 构建统一布局 ----
    plt.rcParams["font.family"] = FONT_FAMILY
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1], hspace=0.28, wspace=0.22)

    # 左上：PCA 散点
    ax_pca = fig.add_subplot(gs[0, 0])
    cl.plot_pca_scatter(assignments, axis=ax_pca)
    _apply_font(ax_pca)

    # 右上：聚类雷达
    ax_radar = fig.add_subplot(gs[0, 1], projection="polar")
    cl.plot_cluster_radar(transformed_profile, axis=ax_radar)
    _apply_font(ax_radar)

    # 下方：转移热力图（跨整行，内部按矩阵数分列）
    n_items = len(transition_matrices)
    gs_bottom = gs[1, :].subgridspec(1, n_items, wspace=0.25)
    heat_axes = [fig.add_subplot(gs_bottom[i]) for i in range(n_items)]
    cl.plot_transition_heatmaps(transition_matrices, axes=heat_axes)
    for hax in heat_axes:
        _apply_font(hax)
        for label in hax.get_xticklabels() + hax.get_yticklabels():
            label.set_fontsize(HEAT_FONT)
        hax.title.set_fontsize(TITLE_SIZE)
        hax.xaxis.label.set_fontsize(LABEL_SIZE)
        hax.yaxis.label.set_fontsize(LABEL_SIZE)

    out_file = args.output_dir / "cluster_profiles_composite.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined figure saved to: {out_file}")


if __name__ == "__main__":
    main()
