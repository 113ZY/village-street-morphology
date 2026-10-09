"""将 Fig2_roc_curves、Fig4_radar、Fig_cm_tabicl 拼成一张严格长方形的组图。

要求：三个子图共同构成一个规整的长方形外轮廓，边缘左右/上下对齐。

布局（2×2 网格，左右边对齐成矩形）：
    +-------------------+-------------------+
    |  Fig2 ROC (左)    |  Fig4 radar (右)  |   上排：左右等宽、贴边(wspace=0)
    +-------------------+-------------------+
    |  Fig_cm_tabicl (横跨整行，与上排左右边对齐) |   下排
    +-------------------+-------------------+

实现要点：
  - 固定 figsize，不用 bbox_inches="tight"，保证输出就是长方形。
  - wspace=0 / hspace 固定，使两排左右边、上下边严格对齐。
  - 每个子图先裁剪白边再统一基准高度缩放，内容尺寸一致。
  - 统一黑色细边框。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image


def _crop_white(img: np.ndarray, pad: int = 6) -> np.ndarray:
    arr = img
    if arr.ndim == 3 and arr.shape[2] == 4:
        mask = arr[:, :, 3] > 8
    else:
        mask = np.any(arr[:, :, :3] < 245, axis=2)
    if not mask.any():
        return arr
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
    r1 = min(arr.shape[0] - 1, r1 + pad)
    c1 = min(arr.shape[1] - 1, c1 + pad)
    return arr[r0:r1 + 1, c0:c1 + 1]


def _resize(img: np.ndarray, base_h: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = int(w * base_h / h)
    return np.asarray(Image.fromarray((img * 255).astype("uint8")).resize((new_w, int(base_h))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine ML figures into one rectangular figure.")
    parser.add_argument("--src-dir", type=Path, default=Path("outputs/LIST橙-蓝-紫"))
    parser.add_argument("--output", type=Path, default=Path("outputs/LIST橙-蓝-紫/ml_figures_composite.png"))
    args = parser.parse_args()

    roc = _resize(_crop_white(plt.imread(args.src_dir / "Fig2_roc_curves.png")), 1100.0)
    radar = _resize(_crop_white(plt.imread(args.src_dir / "Fig4_radar.png")), 1100.0)
    cm = _resize(_crop_white(plt.imread(args.src_dir / "Fig_cm_tabicl.png")), 1100.0)

    # 固定画布尺寸 => 输出严格为长方形
    fig = plt.figure(figsize=(14, 13))
    # 2×2：上排两等宽格，下排横跨；wspace=0 使左右边对齐贴合
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.12, wspace=0.0)

    def _place(ax, img, title):
        ax.imshow(img)
        ax.axis("off")
        for s in ("left", "right", "top", "bottom"):
            ax.spines[s].set_visible(True)
            ax.spines[s].set_linewidth(1.0)
            ax.spines[s].set_color("black")
        ax.set_title(title, fontsize=13, fontfamily="Times New Roman", pad=8)

    ax_roc = fig.add_subplot(gs[0, 0])
    _place(ax_roc, roc, "Fig2  ROC Curves")

    ax_radar = fig.add_subplot(gs[0, 1])
    _place(ax_radar, radar, "Fig4  Cluster Radar")

    ax_cm = fig.add_subplot(gs[1, :])
    _place(ax_cm, cm, "Fig  Confusion Matrix (TabICL)")

    # 固定边距，不用 tight，保证整体为规整长方形
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Combined figure saved to: {args.output}")


if __name__ == "__main__":
    main()
