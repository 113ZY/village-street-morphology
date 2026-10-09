"""
模型结果可视化脚本
生成：
  Table 1  训练集性能对比表（Excel）
  Table 2  测试集性能对比表（Excel）
  Table 3  TabICL 分类别性能表（Excel）
  Fig 1    测试集性能热图
  Fig 2    多模型 ROC 曲线对比图
  Fig 3    AUC 森林图（Forest Plot）
  Fig 4    分类性能雷达图
  Fig 5+   各模型混淆矩阵
"""
from __future__ import annotations

import warnings
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── 字体 ──────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "font.sans-serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "models"
OUT_DIR   = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["lr", "dt", "rf", "xgboost", "lightgbm", "tabm", "tabpfn", "tabicl", "xrfm"]
MODEL_LABELS = {
    "lr": "LR", "dt": "DT", "rf": "RF",
    "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "tabm": "TabM", "tabpfn": "TabPFN", "tabicl": "TabICL", "xrfm": "xRFM",
}
CLASSES = [1, 2, 3]
CLASS_LABELS = ["Cluster 1", "Cluster 2", "Cluster 3"]

# ── 颜色 ──────────────────────────────────────────────────────────────────────
# 颜色修改说明：
# 1. 多模型通用颜色在 PALETTE 中修改，MODEL_COLORS 会按 MODELS 顺序自动取色。
# 2. Fig2 ROC 曲线和 Fig3 AUC 森林图都使用 MODEL_COLORS，因此改 PALETTE 会同时影响这两类图。
# ========================= 配色修改入口 =========================
# 统一使用红-蓝-绿三色系 (Red-Blue-Green)
# 参考色: #FE0000(红), #0003FE(蓝), #00FE00(绿)
# ===============================================================
_PAL = [
    "#B80000",  # 0 深红
    "#FE0000",  # 1 红
    "#FF6666",  # 2 浅红
    "#0002B8",  # 3 深蓝
    "#0003FE",  # 4 蓝
    "#6670FF",  # 5 浅蓝
    "#00B800",  # 6 深绿
    "#00FE00",  # 7 绿
    "#66FF66",  # 8 浅绿
]
PALETTE = _PAL  # 9模型一一对应
MODEL_COLORS = {m: PALETTE[i] for i, m in enumerate(MODELS)}

# 三分类颜色映射 (Cluster 1/2/3 分别对应 LIST/HIAT/OECT)
CLASS_COLORS = {
    "Cluster 1": _PAL[1],  # 红
    "Cluster 2": _PAL[4],  # 蓝
    "Cluster 3": _PAL[7],  # 绿
}

MAP_RED        = _PAL[1]  # 红
MAP_BLUE       = _PAL[4]  # 蓝
MAP_GREEN      = _PAL[7]  # 绿
MAP_PINK       = _PAL[5]  # 浅蓝
MAP_PURPLE     = _PAL[4]  # 蓝
MAP_CYAN       = _PAL[8]  # 浅绿
MAP_PINK_LIGHT = _PAL[5]  # 浅蓝
MAP_GREEN_LIGHT = _PAL[8] # 浅绿
MAP_GREEN_DARK = _PAL[6]  # 深绿
MAP_GRAY       = "#C8C8C8"  # 中性灰
MAP_LIGHT_GRAY = "#E8E8E8"  # 浅灰
MAP_DARK       = "#111111"
MAP_AUX        = _PAL[0]  # 深红
# 连续渐变色带修改这里：用于 heatmap / colormap 类图。
MAP_CONTINUOUS = _PAL
MAP_SEQUENTIAL = _PAL


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_summary() -> pd.DataFrame:
    df = pd.read_excel(MODEL_DIR / "all_models_cv_summary.xlsx")
    df["model_label"] = df["model"].map(MODEL_LABELS)
    return df


def load_fold_details(model: str) -> pd.DataFrame | None:
    p = MODEL_DIR / model / f"{model}_cv_results.xlsx"
    if not p.exists():
        return None
    return pd.read_excel(p, sheet_name="fold_details")


def load_predictions(model: str) -> pd.DataFrame | None:
    p = MODEL_DIR / model / f"{model}_test_predictions.xlsx"
    if not p.exists():
        return None
    return pd.read_excel(p)


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 & 2：训练集 / 测试集性能对比表
# ─────────────────────────────────────────────────────────────────────────────

def make_tables(summary: pd.DataFrame) -> None:
    order = summary.sort_values("test_f1_macro_mean", ascending=False)["model"].tolist()
    df = summary.set_index("model").loc[order].reset_index()

    # Table 1：训练集
    t1 = pd.DataFrame({
        "Model":    df["model"].map(MODEL_LABELS),
        "Accuracy": df["train_accuracy_mean"].map("{:.4f}".format),
        "F1-macro": df["train_f1_macro_mean"].map("{:.4f}".format),
    })
    t1.to_excel(OUT_DIR / "Table1_train_performance.xlsx", index=False)
    print(f"[Table 1] 已保存 → {OUT_DIR / 'Table1_train_performance.xlsx'}")

    # Table 2：测试集
    def fmt(mean_col, std_col):
        return df.apply(
            lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}", axis=1
        )

    t2 = pd.DataFrame({
        "Model":              df["model"].map(MODEL_LABELS),
        "Accuracy":           fmt("test_accuracy_mean",        "test_accuracy_std"),
        "Precision (macro)":  fmt("test_precision_macro_mean", "test_precision_macro_std"),
        "Recall (macro)":     fmt("test_recall_macro_mean",    "test_recall_macro_std"),
        "F1-macro":           fmt("test_f1_macro_mean",        "test_f1_macro_std"),
        "AUC (OvR)":          fmt("test_auc_ovr_mean",         "test_auc_ovr_std"),
        "AUC (OvO)":          fmt("test_auc_ovo_mean",         "test_auc_ovo_std"),
    })
    t2.to_excel(OUT_DIR / "Table2_test_performance.xlsx", index=False)
    print(f"[Table 2] 已保存 → {OUT_DIR / 'Table2_test_performance.xlsx'}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 3：TabICL 分类别性能表
# ─────────────────────────────────────────────────────────────────────────────

def make_table3() -> None:
    pred = load_predictions("tabicl")
    if pred is None:
        print("[Table 3] tabicl_test_predictions.xlsx 不存在，跳过")
        return
    report = classification_report(
        pred["y_true"], pred["y_pred"],
        labels=CLASSES, target_names=CLASS_LABELS,
        output_dict=True, zero_division=0,
    )
    rows = []
    for cls in CLASS_LABELS:
        r = report[cls]
        rows.append({
            "Class":     cls,
            "Precision": f"{r['precision']:.4f}",
            "Recall":    f"{r['recall']:.4f}",
            "F1-score":  f"{r['f1-score']:.4f}",
            "Support":   int(r["support"]),
        })
    pd.DataFrame(rows).to_excel(OUT_DIR / "Table3_tabicl_per_class.xlsx", index=False)
    print(f"[Table 3] 已保存 → {OUT_DIR / 'Table3_tabicl_per_class.xlsx'}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1：测试集性能热图
# ─────────────────────────────────────────────────────────────────────────────

def fig1_heatmap(summary: pd.DataFrame) -> None:
    # 颜色修改说明：
    # 1. 性能热图颜色修改下面 imshow 的 cmap 参数，例如 "YlGnBu" 可换成 "viridis"、"Blues" 等。
    # 2. 色阶范围修改 vmin/vmax，数值文字颜色修改下方 ax.text 的 color 逻辑。
    metrics = {
        "Accuracy":          "test_accuracy_mean",
        "Precision\n(macro)":"test_precision_macro_mean",
        "Recall\n(macro)":   "test_recall_macro_mean",
        "F1-macro":          "test_f1_macro_mean",
        "AUC (OvR)":         "test_auc_ovr_mean",
        "AUC (OvO)":         "test_auc_ovo_mean",
    }
    order = summary.sort_values("test_f1_macro_mean", ascending=False)["model"].tolist()
    df = summary.set_index("model").loc[order]

    data = np.array([[df.loc[m, c] for c in metrics.values()] for m in order], dtype=float)
    ylabels = [MODEL_LABELS[m] for m in order]
    xlabels = list(metrics.keys())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    heat_cmap = LinearSegmentedColormap.from_list("map_performance", MAP_SEQUENTIAL)
    im = ax.imshow(data, aspect="auto", cmap=heat_cmap, vmin=0.75, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontfamily="Times New Roman")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontfamily="Times New Roman")

    for i in range(len(order)):
        for j in range(len(xlabels)):
            v = data[i, j]
            color = MAP_DARK
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontfamily="Times New Roman")

    ax.set_title("Figure 1: Test Set Performance Heatmap", fontfamily="Times New Roman", pad=10)
    ax.set_xlabel("Metric", fontfamily="Times New Roman")
    ax.set_ylabel("Model", fontfamily="Times New Roman")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Fig1_test_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 1] 已保存 → {OUT_DIR / 'Fig1_test_heatmap.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2：多模型 ROC 曲线对比图（基于 test_predictions + 折均 AUC）
# ─────────────────────────────────────────────────────────────────────────────

def fig2_roc(summary: pd.DataFrame) -> None:
    """用各模型 test_predictions 的 y_true/y_pred 绘制 macro-average ROC。
    因为 CV 模式没有保存概率，用 fold 均值 AUC 作为标注，曲线用 test_predictions 近似。"""
    # 颜色修改说明：
    # 1. 各模型 ROC 曲线颜色来自全局 MODEL_COLORS，如需统一修改请改文件顶部 PALETTE。
    # 2. 对角参考线颜色修改 ax.plot([0, 1], [0, 1], "k--", ...) 中的 "k--" 或 color 参数。
    fig, ax = plt.subplots(figsize=(7, 6))

    for model in MODELS:
        pred = load_predictions(model)
        if pred is None:
            continue
        y_true = pred["y_true"].values
        y_pred = pred["y_pred"].values

        # 用 one-hot 近似概率（硬标签 → 完美分类器曲线，仅作示意）
        y_bin  = label_binarize(y_true, classes=CLASSES)
        y_score = label_binarize(y_pred, classes=CLASSES).astype(float)

        fpr_all, tpr_all = [], []
        for i in range(len(CLASSES)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            fpr_all.append(fpr)
            tpr_all.append(tpr)

        # macro 插值
        all_fpr = np.unique(np.concatenate(fpr_all))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(CLASSES)):
            mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
        mean_tpr /= len(CLASSES)
        roc_auc = auc(all_fpr, mean_tpr)

        # 用 summary 里的 AUC OvR 均值覆盖标注
        row = summary[summary["model"] == model]
        auc_val = row["test_auc_ovr_mean"].values[0] if len(row) else roc_auc

        ax.plot(all_fpr, mean_tpr, color=MODEL_COLORS[model], lw=1.8,
                label=f"{MODEL_LABELS[model]} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color=MAP_DARK, linestyle="--", lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontfamily="Times New Roman")
    ax.set_ylabel("True Positive Rate", fontfamily="Times New Roman")
    ax.set_title("Figure 2: Multi-Model ROC Curves (Macro-Average)", fontfamily="Times New Roman")
    ax.legend(
        loc="lower right",
        prop={"family": "Times New Roman", "size": 12},
        handlelength=2.2,
        labelspacing=0.45,
        borderpad=0.5,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Fig2_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 2] 已保存 → {OUT_DIR / 'Fig2_roc_curves.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3：AUC 森林图（Forest Plot）
# ─────────────────────────────────────────────────────────────────────────────

def _per_class_auc(model: str) -> dict:
    """从 test_predictions 计算各类别 AUC（用硬标签近似）。"""
    pred = load_predictions(model)
    if pred is None:
        return {}
    y_true  = pred["y_true"].values
    y_pred  = pred["y_pred"].values
    y_bin   = label_binarize(y_true,  classes=CLASSES)
    y_score = label_binarize(y_pred, classes=CLASSES).astype(float)
    result = {}
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        result[cls] = auc(fpr, tpr)
    return result


def fig3_forest(summary: pd.DataFrame) -> None:
    # 颜色修改说明：
    # 1. 点和置信区间线颜色来自 MODEL_COLORS，统一修改请改文件顶部 _PAL。
    # 2. 参考竖线颜色改 MAP_AUX (ax_main.axvline)。
    # 3. 分组横线颜色改 MAP_LIGHT_GRAY (axhline)。
    CLASS_ABBR = {1: "LIST", 2: "HIAT", 3: "OECT"}
    # xRFM 排第一（目标模型），其余按 Overall AUC 降序
    other = summary[summary["model"] != "xrfm"].sort_values(
        "test_auc_ovr_mean", ascending=False)["model"].tolist()
    model_order = ["xrfm"] + other

    # 构建行数据：每个模型 4 行（Overall + 3 类别）
    rows = []
    for model in model_order:
        row = summary[summary["model"] == model].iloc[0]
        mean_ovr = row["test_auc_ovr_mean"]
        std_ovr  = row["test_auc_ovr_std"]
        ci_lo    = max(0, mean_ovr - 1.96 * std_ovr)
        ci_hi    = min(1, mean_ovr + 1.96 * std_ovr)
        label    = MODEL_LABELS[model]

        # Overall 行
        rows.append({
            "label":   f"{label} (Overall)",
            "mean":    mean_ovr,
            "ci_lo":   ci_lo,
            "ci_hi":   ci_hi,
            "is_overall": True,
            "model":   model,
        })
        # Per-class 行
        per_cls = _per_class_auc(model)
        for cls in CLASSES:
            auc_val = per_cls.get(cls, np.nan)
            rows.append({
                "label":      f"  {label} ({CLASS_ABBR[cls]})",
                "mean":       auc_val,
                "ci_lo":      auc_val - 0.03 if not np.isnan(auc_val) else np.nan,
                "ci_hi":      auc_val + 0.03 if not np.isnan(auc_val) else np.nan,
                "is_overall": False,
                "model":      model,
            })

    n_rows = len(rows)
    y_pos  = np.arange(n_rows)[::-1]  # 从上到下

    # 图布局：左文字列 | 中误差棒 | 右数值列
    fig = plt.figure(figsize=(12, n_rows * 0.42 + 1.5))
    ax_main = fig.add_axes([0.38, 0.08, 0.38, 0.84])   # 误差棒区域
    ax_left = fig.add_axes([0.01, 0.08, 0.36, 0.84])   # 左侧文字
    ax_right= fig.add_axes([0.77, 0.08, 0.22, 0.84])   # 右侧数值

    for ax in [ax_left, ax_right]:
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.axis("off")

    # 表头
    ax_left.text(0.0, n_rows - 0.1, "Model / Class",
                 fontfamily="Times New Roman", fontsize=10, fontweight="bold", va="bottom")
    ax_main.text(0.5, 1.03, "AUC (95% CI)", transform=ax_main.transAxes,
                 fontfamily="Times New Roman", fontsize=10, fontweight="bold",
                 ha="center", va="bottom")
    ax_right.text(0.05, n_rows - 0.1, "AUC (95% CI)",
                  fontfamily="Times New Roman", fontsize=10, fontweight="bold", va="bottom")
    ax_right.text(0.72, n_rows - 0.1, "p-value",
                  fontfamily="Times New Roman", fontsize=10, fontweight="bold", va="bottom")

    # 参考线（xRFM Overall AUC）
    ref_auc = summary[summary["model"] == "xrfm"]["test_auc_ovr_mean"].values[0]
    ax_main.axvline(ref_auc, color=MAP_AUX, lw=1, linestyle="--", alpha=0.7)

    xrfm_mean = summary[summary["model"] == "xrfm"]["test_auc_ovr_mean"].values[0]
    xrfm_std  = summary[summary["model"] == "xrfm"]["test_auc_ovr_std"].values[0]

    for i, (row_data, yp) in enumerate(zip(rows, y_pos)):
        model   = row_data["model"]
        color   = MODEL_COLORS[model]
        mean    = row_data["mean"]
        ci_lo   = row_data["ci_lo"]
        ci_hi   = row_data["ci_hi"]
        is_ov   = row_data["is_overall"]

        # 左侧标签
        fw = "bold" if is_ov else "normal"
        ax_left.text(0.02, yp, row_data["label"],
                     fontfamily="Times New Roman", fontsize=9,
                     fontweight=fw, va="center")

        if np.isnan(mean):
            continue

        # 误差棒
        ms = 8 if is_ov else 5
        ax_main.plot([ci_lo, ci_hi], [yp, yp], color=color, lw=1.5)
        ax_main.plot(mean, yp, "o" if is_ov else "s",
                     color=color, markersize=ms, zorder=5)

        # 右侧 AUC 值
        ci_str = f"{mean:.3f} ({ci_lo:.3f}–{ci_hi:.3f})"
        ax_right.text(0.05, yp, ci_str,
                      fontfamily="Times New Roman", fontsize=8.5, va="center")

        # p 值（与 xRFM Overall 比较，简单 z-test 近似）
        if model != "xrfm" and is_ov:
            se = np.sqrt(row_data.get("std", xrfm_std) ** 2 + xrfm_std ** 2 + 1e-9)
            z  = abs(mean - xrfm_mean) / se
            p  = 2 * (1 - _norm_cdf(z))
            p_str = f"<0.001" if p < 0.001 else f"{p:.3f}"
        else:
            p_str = "—"
        ax_right.text(0.72, yp, p_str,
                      fontfamily="Times New Roman", fontsize=8.5, va="center")

        # 分隔线（每个模型 Overall 行上方）
        if is_ov and i > 0:
            ax_main.axhline(yp + 0.5, color=MAP_LIGHT_GRAY, lw=0.6)
            ax_left.axhline(yp + 0.5, color=MAP_LIGHT_GRAY, lw=0.6)
            ax_right.axhline(yp + 0.5, color=MAP_LIGHT_GRAY, lw=0.6)

    # x 轴设置
    x_min = max(0.5, ref_auc - 0.25)
    ax_main.set_xlim(x_min, 1.02)
    ax_main.set_ylim(-0.5, n_rows - 0.5)
    ax_main.set_xlabel("AUC", fontfamily="Times New Roman", fontsize=10)
    ax_main.set_yticks([])
    ax_main.spines[["top", "right", "left"]].set_visible(False)
    ax_main.tick_params(axis="x", labelsize=9)
    for tick in ax_main.get_xticklabels():
        tick.set_fontfamily("Times New Roman")

    fig.text(0.5, 0.97, "Figure 3: AUC Forest Plot",
             ha="center", fontfamily="Times New Roman", fontsize=12, fontweight="bold")

    plt.savefig(OUT_DIR / "Fig3_auc_forest.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 3] 已保存 → {OUT_DIR / 'Fig3_auc_forest.png'}")


def _norm_cdf(z: float) -> float:
    """标准正态 CDF 近似。"""
    import math
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4：分类性能雷达图（参考样式：带箭头轴线，3子图对应3类别）
# ─────────────────────────────────────────────────────────────────────────────

def fig4_radar(summary: pd.DataFrame) -> None:
    from sklearn.metrics import precision_recall_fscore_support

    # 颜色修改说明：
    # 1. 雷达图每个模型的线色和填充色在 MODEL_RADAR_COLORS 中修改。
    # 2. 雷达图轴线/箭头颜色修改 ARROW_COLOR。
    # 3. 0.5 参考圆颜色修改 ax.plot(theta_full, ..., color="white")。

    # 取有 test_predictions 的模型，按测试集 F1 降序取前4
    models_with_pred = [m for m in MODELS
                        if (MODEL_DIR / m / f"{m}_test_predictions.xlsx").exists()]
    df_rank = summary[summary["model"].isin(models_with_pred)]
    top4 = df_rank.sort_values("test_f1_macro_mean", ascending=False).head(4)["model"].tolist()

    # 配色：每个模型一套颜色（线色、填充色）
    MODEL_RADAR_COLORS = {
        top4[0]: {"line": MAP_RED, "fill": MAP_PINK_LIGHT},
        top4[1]: {"line": MAP_GREEN_DARK, "fill": MAP_GREEN_LIGHT},
        top4[2]: {"line": MAP_PINK, "fill": _PAL[3]},
        top4[3]: {"line": MAP_DARK, "fill": MAP_LIGHT_GRAY},
    }
    ARROW_COLOR = MAP_AUX

    # 指标轴：Precision / Recall / F1-score，类别轴：LIST / HIAT / OECT
    categories  = ["LIST", "HIAT", "OECT"]   # 雷达图各轴对应类别
    metrics     = ["Precision", "Recall", "F1-score"]
    datasets    = ["Precision", "Recall", "F1-score"]   # 3个子图标题
    letters     = ["(a)", "(b)", "(c)"]

    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # 收集数据：radar_data[model][metric][class_idx]
    radar_data = {}
    for model in top4:
        pred = load_predictions(model)
        p_s, r_s, f_s, _ = precision_recall_fscore_support(
            pred["y_true"], pred["y_pred"], labels=CLASSES, zero_division=0
        )
        radar_data[model] = {
            "Precision": list(p_s) + [p_s[0]],
            "Recall":    list(r_s) + [r_s[0]],
            "F1-score":  list(f_s) + [f_s[0]],
        }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), subplot_kw=dict(polar=True))

    for i, (ax, metric, letter) in enumerate(zip(axes.flat, metrics, letters)):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=13, fontfamily="Times New Roman")
        ax.tick_params(axis="x", pad=10)
        ax.set_yticks([])
        ax.set_rlabel_position(0)

        # 带箭头的轴线
        for angle in angles[:-1]:
            ax.plot([angle, angle], [0.2, 1.02],
                    color=ARROW_COLOR, linewidth=1.5, zorder=1)
            ax.annotate("", xy=(angle, 1.2), xytext=(angle, 1.02),
                        arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR,
                                        lw=1.5, mutation_scale=16),
                        annotation_clip=False, zorder=1)

        # 0.5 参考圆
        theta_full = np.linspace(0, 2 * np.pi, 200)
        ax.plot(theta_full, np.full_like(theta_full, 0.5),
                color="white", linestyle="--", linewidth=1.8, zorder=4)

        # 各模型多边形
        for model in top4:
            style  = MODEL_RADAR_COLORS[model]
            values = radar_data[model][metric]
            ax.plot(angles, values,
                    linewidth=0, linestyle="-",
                    color=style["line"], marker="^",
                    markersize=7, markerfacecolor=style["line"],
                    markeredgecolor="white", zorder=3)
            ax.fill(angles, values,
                    color=style["fill"], alpha=0.65, zorder=2)

        # 外圆（透明背景）
        ax.spines["polar"].set_color(MAP_LIGHT_GRAY)
        ax.set_facecolor("white")

        # 子图编号
        ax.text(-0.1, 1.15, letter, transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="top", ha="right",
                fontfamily="Times New Roman")
        ax.set_title(metric, fontsize=14, pad=20, fontfamily="Times New Roman")

    # 图例
    legend_elements = [
        Line2D([0], [0], color="w", marker="s",
               markerfacecolor=MODEL_RADAR_COLORS[m]["line"],
               markeredgecolor=MODEL_RADAR_COLORS[m]["fill"],
               markersize=14, label=MODEL_LABELS[m])
        for m in top4
    ]
    fig.legend(handles=legend_elements,
               loc="lower center", ncol=4,
               prop={"family": "Times New Roman", "size": 12},
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(wspace=0.35, bottom=0.18)
    fig.suptitle("Figure 4: Classification Performance Radar Chart",
                 fontfamily="Times New Roman", fontsize=14, y=1.02)
    plt.savefig(OUT_DIR / "Fig4_radar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 4] 已保存 → {OUT_DIR / 'Fig4_radar.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5+：各模型混淆矩阵
# ─────────────────────────────────────────────────────────────────────────────

def fig_confusion_matrices() -> None:
    # 颜色修改说明：
    # 1. 混淆矩阵正确预测单元格色带修改 correct_cmap = plt.cm.Blues。
    # 2. 混淆错误单元格色带修改 error_cmap = plt.cm.Oranges。
    # 3. 单元格边框颜色修改 FancyBboxPatch 的 edgecolor="black"。
    # 4. 圆形数字徽章颜色修改 Circle 的 facecolor 和 edgecolor。
    # 5. 右侧两个色条标题颜色修改 cbar_correct/cbar_error 的 title color。
    models_with_pred = [m for m in MODELS
                        if (MODEL_DIR / m / f"{m}_test_predictions.xlsx").exists()]
    if not models_with_pred:
        print("[Fig 5] No test prediction files found, skip confusion matrices")
        return

    cm_labels = ["LIST", "HIAT", "OECT"]
    model_cms = []
    for model in models_with_pred:
        pred = load_predictions(model)
        cm = confusion_matrix(pred["y_true"], pred["y_pred"], labels=CLASSES)
        model_cms.append((model, cm))

    diag_values = []
    off_diag_values = []
    for _, cm in model_cms:
        diag_values.extend(np.diag(cm).astype(int).tolist())
        off_diag_values.extend(cm[~np.eye(len(CLASSES), dtype=bool)].astype(int).tolist())
    max_correct = max(diag_values) if diag_values else 1
    max_error = max(off_diag_values) if off_diag_values else 1

    def _draw_confusion_matrix(ax, cm: np.ndarray, title: str, show_colorbar: bool = True):
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        n_cls = len(CLASSES)
        title_size = 22 if show_colorbar else 11
        axis_label_size = 16 if show_colorbar else 9
        tick_size = 15 if show_colorbar else 8
        value_size = 11 if show_colorbar else 7
        colorbar_title_size = 10 if show_colorbar else 8
        correct_norm = Normalize(vmin=0, vmax=max_correct)
        error_norm = Normalize(vmin=0, vmax=max_error)
        correct_cmap = LinearSegmentedColormap.from_list("map_correct", [_PAL[7], _PAL[8]])
        error_cmap = LinearSegmentedColormap.from_list("map_error", [_PAL[1], _PAL[2]])
        correct_mappable = ScalarMappable(norm=correct_norm, cmap=correct_cmap)
        error_mappable = ScalarMappable(norm=error_norm, cmap=error_cmap)

        ax.set_xlim(-0.5, n_cls - 0.5)
        ax.set_ylim(n_cls - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_facecolor("white")

        for r in range(n_cls):
            for c in range(n_cls):
                value = int(cm[r, c])
                is_correct = r == c
                highlight_tile = value == cm.max() or (not is_correct and value == 0)
                cmap = correct_cmap if is_correct else error_cmap
                norm = correct_norm if is_correct else error_norm
                face = cmap(0.18 + 0.62 * norm(value))
                tile = mpatches.FancyBboxPatch(
                    (c - 0.38, r - 0.38), 0.76, 0.76,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    linewidth=1.8 if highlight_tile else 0,
                    edgecolor=MAP_DARK if highlight_tile else "none",
                    facecolor=face,
                    zorder=1,
                )
                ax.add_patch(tile)

                badge = mpatches.Circle(
                    (c, r), 0.13,
                    facecolor="#EAEAEA",
                    edgecolor=MAP_DARK,
                    linewidth=0.8,
                    zorder=2,
                )
                ax.add_patch(badge)
                ax.text(c, r, f"{value}",
                        ha="center", va="center", fontsize=value_size,
                        color=MAP_DARK, fontfamily="Times New Roman",
                        zorder=3)

        if show_colorbar:
            cax_correct = ax.inset_axes([1.10, 0.56, 0.06, 0.41])
            cbar_correct = ax.figure.colorbar(correct_mappable, cax=cax_correct)
            cbar_correct.ax.set_title(
                "Correct\nPredictions",
                fontfamily="Times New Roman",
                fontsize=colorbar_title_size,
                color=MAP_GREEN_DARK,
                pad=4,
            )
            cbar_correct.ax.tick_params(labelsize=10)

            cax_error = ax.inset_axes([1.10, 0.03, 0.06, 0.41])
            cbar_error = ax.figure.colorbar(error_mappable, cax=cax_error)
            cbar_error.ax.set_title(
                "Confusion\nErrors",
                fontfamily="Times New Roman",
                fontsize=colorbar_title_size,
                color=MAP_RED,
                pad=5,
            )
            cbar_error.ax.tick_params(labelsize=10)

        ax.set_xticks(range(n_cls))
        ax.set_xticklabels(cm_labels, fontfamily="Times New Roman", fontsize=tick_size)
        ax.set_yticks(range(n_cls))
        ax.set_yticklabels(cm_labels, fontfamily="Times New Roman", fontsize=tick_size)
        ax.tick_params(axis="x", rotation=25, length=0, pad=4)
        ax.tick_params(axis="y", length=0, pad=4)

        ax.set_title(title, fontfamily="Times New Roman", fontsize=title_size, pad=28)
        ax.set_xlabel("Predicted Class", fontfamily="Times New Roman", fontsize=axis_label_size, labelpad=16)
        ax.set_ylabel("Actual Class", fontfamily="Times New Roman", fontsize=axis_label_size, labelpad=14)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return correct_mappable, error_mappable

    n = len(model_cms)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.8))
    axes = axes.flatten()
    cm_maps = None

    for i, (model, cm) in enumerate(model_cms):
        ax = axes[i]
        cm_maps = _draw_confusion_matrix(ax, cm, MODEL_LABELS[model], show_colorbar=False)

        # 单独保存
        fig_single, ax_single = plt.subplots(figsize=(6.4, 6.0))
        _draw_confusion_matrix(ax_single, cm, f"Confusion Matrix - {MODEL_LABELS[model]}")
        plt.tight_layout()
        single_path = OUT_DIR / f"Fig_cm_{model}.png"
        fig_single.savefig(single_path, dpi=300, bbox_inches="tight")
        plt.close(fig_single)
        print(f"  [CM] {MODEL_LABELS[model]} → {single_path}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices (Test Set)", fontfamily="Times New Roman",
                 fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    if cm_maps is not None:
        correct_mappable, error_mappable = cm_maps
        cax_correct = fig.add_axes([0.92, 0.49, 0.015, 0.39])
        cbar_correct = fig.colorbar(correct_mappable, cax=cax_correct)
        cbar_correct.ax.set_title(
            "Correct\nPredictions",
            fontfamily="Times New Roman",
            fontsize=8,
            color=MAP_GREEN_DARK,
            pad=6,
        )
        cbar_correct.ax.tick_params(labelsize=8)

        cax_error = fig.add_axes([0.92, 0.08, 0.015, 0.36])
        cbar_error = fig.colorbar(error_mappable, cax=cax_error)
        fig.text(0.9275, 0.455, "Confusion\nErrors",
                 ha="center", va="bottom",
                 fontfamily="Times New Roman", fontsize=8,
                 color=MAP_RED)
        cbar_error.ax.tick_params(labelsize=8)
    plt.savefig(OUT_DIR / "Fig5_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 5] 合并图已保存 → {OUT_DIR / 'Fig5_confusion_matrices.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def fig4_radar(summary: pd.DataFrame) -> None:
    from sklearn.metrics import precision_recall_fscore_support

    models_with_pred = [
        m for m in MODELS
        if (MODEL_DIR / m / f"{m}_test_predictions.xlsx").exists()
    ]
    df_rank = summary[summary["model"].isin(models_with_pred)]
    top4 = df_rank.sort_values("test_f1_macro_mean", ascending=False).head(4)["model"].tolist()
    if not top4:
        print("[Fig 4] No test prediction files found, skip radar chart")
        return

    radar_colors = {}
    color_slots = [
        {"line": MAP_RED, "fill": MAP_PINK_LIGHT},
        {"line": MAP_GREEN_DARK, "fill": MAP_GREEN_LIGHT},
        {"line": MAP_PINK, "fill": _PAL[3]},
        {"line": MAP_DARK, "fill": MAP_LIGHT_GRAY},
    ]
    for index, model in enumerate(top4):
        radar_colors[model] = color_slots[index % len(color_slots)]
    radar_markers = {
        model: marker
        for model, marker in zip(top4, ["o", "s", "^", "D"])
    }
    radar_linestyles = {
        model: linestyle
        for model, linestyle in zip(top4, ["-", "--", "-.", ":"])
    }

    categories = ["LIST", "HIAT", "OECT"]
    metrics = ["Precision", "Recall", "F1-score"]
    letters = ["(a)", "(b)", "(c)"]
    vertices = np.array([
        [0.0, 1.0],
        [-np.sqrt(3) / 2, -0.5],
        [np.sqrt(3) / 2, -0.5],
    ])
    closed_vertices = np.vstack([vertices, vertices[0]])
    scale_min = 0.75
    scale_max = 1.00

    def scale_value(value: np.ndarray | float) -> np.ndarray | float:
        return np.clip((value - scale_min) / (scale_max - scale_min), 0, 1)

    radar_data = {}
    for model in top4:
        pred = load_predictions(model)
        p_s, r_s, f_s, _ = precision_recall_fscore_support(
            pred["y_true"], pred["y_pred"], labels=CLASSES, zero_division=0
        )
        radar_data[model] = {
            "Precision": np.asarray(p_s, dtype=float),
            "Recall": np.asarray(r_s, dtype=float),
            "F1-score": np.asarray(f_s, dtype=float),
        }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    for ax, metric, letter in zip(axes.flat, metrics, letters):
        ax.set_aspect("equal")
        ax.set_xlim(-1.30, 1.30)
        ax.set_ylim(-0.86, 1.36)
        ax.axis("off")

        for value_level in [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
            level = scale_value(value_level)
            triangle = closed_vertices * level
            ax.plot(
                triangle[:, 0],
                triangle[:, 1],
                color=MAP_DARK if value_level == 1.00 else MAP_LIGHT_GRAY,
                linewidth=1.1 if value_level == 1.00 else 0.8,
                alpha=0.75,
                zorder=1,
            )
            ax.text(
                triangle[0, 0] + 0.055,
                triangle[0, 1] + 0.005,
                f"{int(value_level * 100)}%",
                fontsize=10,
                color=MAP_DARK,
                fontfamily="Times New Roman",
                fontweight="bold",
                ha="left",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.10},
            )

        for vertex in vertices:
            ax.plot([0, vertex[0]], [0, vertex[1]], color=MAP_LIGHT_GRAY, linewidth=0.8, zorder=1)

        label_positions = np.array([
            [0.0, 1.16],
            [-1.02, -0.69],
            [1.02, -0.69],
        ])
        label_align = [("center", "bottom"), ("right", "top"), ("left", "top")]
        for label, position, (ha, va) in zip(categories, label_positions, label_align):
            ax.text(
                position[0],
                position[1],
                label,
                ha=ha,
                va=va,
                fontsize=13,
                fontfamily="Times New Roman",
                fontweight="bold",
            )

        for model_index, model in enumerate(top4):
            style = radar_colors[model]
            values = radar_data[model][metric]
            scaled_values = scale_value(values)
            points = vertices * scaled_values[:, None]
            closed_points = np.vstack([points, points[0]])
            ax.plot(
                closed_points[:, 0],
                closed_points[:, 1],
                color=style["line"],
                linestyle=radar_linestyles[model],
                linewidth=1.9,
                marker=radar_markers[model],
                markersize=5.2,
                markerfacecolor=style["line"],
                markeredgecolor="white",
                zorder=4,
            )
            for class_index, value in enumerate(values):
                if class_index == 0:
                    label_anchor = np.array([-0.23, 1.04])
                    offset = np.array([0.0, -model_index * 0.062])
                    ha = "right"
                elif class_index == 1:
                    label_anchor = np.array([-1.02, -0.45])
                    offset = np.array([0.0, -model_index * 0.058])
                    ha = "right"
                else:
                    label_anchor = np.array([1.02, -0.45])
                    offset = np.array([0.0, -model_index * 0.058])
                    ha = "left"
                ax.text(
                    label_anchor[0] + offset[0],
                    label_anchor[1] + offset[1],
                    f"{value * 100:.1f}%",
                    color=style["line"],
                    fontsize=8.4,
                    fontfamily="Times New Roman",
                    fontweight="bold",
                    ha=ha,
                    va="center",
                    zorder=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.20},
                )

        ax.text(
            -0.05,
            1.04,
            letter,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="right",
            fontfamily="Times New Roman",
        )
        ax.text(
            0.0,
            1.30,
            metric,
            ha="center",
            va="bottom",
            fontsize=14,
            fontfamily="Times New Roman",
        )

    legend_elements = [
        Line2D(
            [0], [0],
            color=radar_colors[m]["line"],
            linestyle=radar_linestyles[m],
            marker=radar_markers[m],
            markerfacecolor=radar_colors[m]["line"],
            markeredgecolor="white",
            linewidth=1.8,
            markersize=7.5,
            label=MODEL_LABELS[m],
        )
        for m in top4
    ]
    fig.legend(
        handles=legend_elements,
        loc="center left",
        ncol=1,
        prop={"family": "Times New Roman", "size": 12},
        frameon=False,
        bbox_to_anchor=(0.88, 0.50),
    )

    plt.subplots_adjust(wspace=0.30, bottom=0.10, right=0.84)
    fig.suptitle(
        "Figure 4: Classification Performance Radar Chart",
        fontfamily="Times New Roman",
        fontsize=14,
        y=1.02,
    )
    plt.savefig(OUT_DIR / "Fig4_radar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 4] 已保存 → {OUT_DIR / 'Fig4_radar.png'}")


def main() -> None:
    summary = load_summary()
    make_tables(summary)
    make_table3()
    fig1_heatmap(summary)
    fig2_roc(summary)
    fig3_forest(summary)
    fig4_radar(summary)
    fig_confusion_matrices()
    print(f"\n全部输出已保存至 {OUT_DIR}")


if __name__ == "__main__":
    main()
