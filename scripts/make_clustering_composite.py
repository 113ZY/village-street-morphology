"""将彩色 IQR 效应图、PCA 方差解释、K 选择曲线拼成一张长方形组图。

布局：
    +---------------------------+
    |   IQR winsorization effect |   (上，跨两列，每指标彩色箱线图)
    +-------------+-------------+
    | PCA variance | K-selection |   (左下 / 右下，统一字体重绘)
    +-------------+-------------+

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
LEGEND_SIZE = 11


def _apply_font(ax):
    ax.title.set_fontfamily(FONT_FAMILY)
    ax.title.set_fontsize(TITLE_SIZE)
    ax.xaxis.label.set_fontfamily(FONT_FAMILY)
    ax.xaxis.label.set_fontsize(LABEL_SIZE)
    ax.yaxis.label.set_fontfamily(FONT_FAMILY)
    ax.yaxis.label.set_fontsize(LABEL_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)
        label.set_fontsize(TICK_SIZE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the combined clustering figure.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/clustering"),
    )
    args = parser.parse_args()

    # ---- 复用 pipeline 的前处理与聚类链路 ----
    df = cl.read_morphology_data(None, cl.Path("data"))
    original_features = cl.prepare_input_features(df)
    transformed_features = cl.apply_iqr_winsorization(original_features)

    _labels, _metrics, _loadings, pca_features, _pca, _model, _scaler, pca_variance = (
        cl.run_fixed_clustering(transformed_features, 3, args.random_state)
    )
    k_selection = cl.evaluate_k_selection(
        _scaler.transform(transformed_features),
        pca_features,
        args.random_state,
        cl.K_SELECTION_RANGE,
    )

    # ---- 构建统一布局 ----
    plt.rcParams["font.family"] = FONT_FAMILY
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.28, wspace=0.22)

    # 顶部：彩色 IQR 效应（跨两列，内部再分 2 个箱线子图，每指标一色）
    gs_top = gs[0, :].subgridspec(1, 2, wspace=0.22)
    ax_iqr_l = fig.add_subplot(gs_top[0])
    ax_iqr_r = fig.add_subplot(gs_top[1])
    cl.plot_iqr_winsor_effect(original_features, transformed_features, axes=[ax_iqr_l, ax_iqr_r])
    _apply_font(ax_iqr_l)
    _apply_font(ax_iqr_r)

    # 左下：PCA 方差解释
    ax_pca = fig.add_subplot(gs[1, 0])
    cl.plot_pca_variance(pca_variance, axis=ax_pca)
    _apply_font(ax_pca)

    # 右下：K 选择曲线（双 y 轴）
    ax_k = fig.add_subplot(gs[1, 1])
    cl.plot_k_selection_curve(k_selection, ax=ax_k)
    _apply_font(ax_k)
    # 第二轴的刻度字体也统一
    for ax_twin in ax_k.figure.axes:
        if ax_twin is not ax_k:
            for label in ax_twin.get_yticklabels():
                label.set_fontfamily(FONT_FAMILY)
                label.set_fontsize(TICK_SIZE)
            ax_twin.yaxis.label.set_fontfamily(FONT_FAMILY)
            ax_twin.yaxis.label.set_fontsize(LABEL_SIZE)

    out_file = args.output_dir / "clustering_composite.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined figure saved to: {out_file}")


if __name__ == "__main__":
    main()
