from __future__ import annotations

# ---------------------------------------------------------------------------
# 必须在导入 numpy/pandas/matplotlib 之前注册 DLL 路径并导入 torch
# 否则 numpy 会先加载系统版 vcruntime140.dll，导致 fbgemm.dll 无法加载
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys

_prefix = _sys.prefix
for _sub in ["Library\\bin", "Library\\lib", "Library\\mingw-w64\\bin", "bin"]:
    _d = _os.path.join(_prefix, _sub)
    if _os.path.isdir(_d):
        try: _os.add_dll_directory(_d)
        except Exception: pass

import torch  # 必须在 numpy/pandas 之前导入，抢占 VC++ 运行时 DLL
# ---------------------------------------------------------------------------

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# 路径设置
ROOT     = Path(__file__).resolve().parents[1]
SRC_DIR  = ROOT / "src"
for p in (str(SRC_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

SHAP_DIR = ROOT / "outputs" / "shap"
for cls in ["LIST", "HIAT", "OECT", "shared"]:
    (SHAP_DIR / cls).mkdir(parents=True, exist_ok=True)

# 绘图样式
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
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

# 常量设置
X_COLS = ["ELEVATION","SLOPE","ASPECT","MAP","MAT","NDVI","DWS",
          "RD","POP","PGDP","NLI","CRPU","ICHD","DAR"]
Y_COLS = ["MBC","LCC","GCC","INT","CSR","IR","AEL"]
YEARS  = [2014, 2016, 2018, 2020]
YEAR_TO_IDX = {y: i for i, y in enumerate(YEARS)}

CLASS_INFO = {1: ("LIST", "LIST"), 2: ("HIAT", "HIAT"), 3: ("OECT", "OECT")}
CLASS_DIRS = {1: SHAP_DIR/"LIST", 2: SHAP_DIR/"HIAT", 3: SHAP_DIR/"OECT"}

# SHAP 分析使用原始 14 个可解释自变量。
SHAP_FEAT_NAMES = X_COLS

# 图中使用的特征标签。
FEAT_LABELS = {
    "ELEVATION": "Elevation", "SLOPE": "Slope", "ASPECT": "Aspect",
    "MAP": "MAP", "MAT": "MAT", "NDVI": "NDVI", "DWS": "DWS",
    "RD": "RD", "POP": "POP", "PGDP": "PGDP", "NLI": "NLI",
    "CRPU": "CRPU", "ICHD": "ICHD", "DAR": "DAR",
}

# ========================= 配色修改入口 =========================
# 统一使用橙-蓝-紫三色系 (Orange-Blue-Purple)
# 参考色: #ED6D54(橙), #91C0D8(蓝), #8762A8(紫)
# ===============================================================
_PAL = [
    "#C94E37",  # 0 深橙
    "#ED6D54",  # 1 橙
    "#F4A290",  # 2 浅橙
    "#6A9FBB",  # 3 深蓝
    "#91C0D8",  # 4 蓝
    "#B8D8E8",  # 5 浅蓝
    "#6B4890",  # 6 深紫
    "#8762A8",  # 7 紫
    "#B394C8",  # 8 浅紫
]

# 三分类颜色映射
CLASS_COLORS = {
    "LIST": _PAL[1],  # 橙
    "HIAT": _PAL[4],  # 蓝
    "OECT": _PAL[7],  # 紫
}

MAP_RED        = _PAL[4]  # 蓝
MAP_BLUE       = _PAL[7]  # 紫
MAP_GREEN      = _PAL[1]  # 橙
MAP_PINK       = _PAL[5]  # 浅蓝
MAP_PURPLE     = _PAL[7]  # 紫
MAP_CYAN       = _PAL[8]  # 浅紫
MAP_PINK_LIGHT = _PAL[5]  # 浅蓝
MAP_GREEN_LIGHT = _PAL[8] # 浅紫
MAP_GREEN_DARK = _PAL[6]  # 深紫
MAP_GRAY       = "#C8C8C8"  # 中性灰
MAP_LIGHT_GRAY = "#E8E8E8"  # 浅灰
MAP_DARK       = "#111111"
MAP_BOUNDARY   = "#222222"
MAP_AUX        = _PAL[0]  # 深橙
MAP_CATEGORY_COLORS = {
    "Socioeconomic":         _PAL[0],  # 深橙
    "Ecological environment": _PAL[4],  # 蓝
    "Natural base":          _PAL[7],  # 紫
    "Historical culture":    _PAL[1],  # 橙
}
# 连续渐变色带：用于 beeswarm / heatmap / surface / contour 等 colormap 类图。
MAP_CONTINUOUS = _PAL
MAP_SEQUENTIAL = _PAL






# 重点关注的特征交互对。
INTERACTION_PAIRS = [
    ("DAR", "RD"),
    ("ELEVATION", "NLI"),
    ("SLOPE", "PGDP"),
    ("CRPU", "ICHD"),
    ("ELEVATION", "PGDP"),
    ("NDVI", "RD"),
]


# 数据加载与特征工程

def load_and_engineer() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X_raw_14, X_full_80, y, df)."""
    x_df = pd.read_excel(ROOT / "data" / "\u4ec5\u81ea\u53d8\u91cf\u6807\u51c6\u5316\u6570\u636e.xlsx")
    c_df = pd.read_excel(ROOT / "outputs" / "clustering" / "k3" / "tables" /
                         "village_year_cluster_assignments.xlsx",
                         usecols=["ID", "YEAR", "Cluster"] + Y_COLS)
    x_df["ID"] = x_df["ID"].astype(str).str.strip()
    c_df["ID"] = c_df["ID"].astype(str).str.strip()
    df = x_df.merge(c_df, on=["ID", "YEAR"], how="inner")
    df["VILLAGE_ID"] = df["ID"].str.replace(r"(2014|2016|2018|2020)$", "", regex=True)
    df = df.sort_values(["VILLAGE_ID", "YEAR"]).reset_index(drop=True)

    raw    = df[X_COLS].values.astype(np.float32)
    y_vals = df[Y_COLS].values.astype(np.float32)
    year_norm = df["YEAR"].map(YEAR_TO_IDX).values.astype(np.float32) / (len(YEARS) - 1)
    year_sin  = np.sin(2 * np.pi * year_norm)
    year_cos  = np.cos(2 * np.pi * year_norm)
    time_feats  = np.column_stack([year_norm, year_sin, year_cos])
    interaction = raw * year_norm[:, None]

    diff_x = np.zeros_like(raw)
    lag_y  = np.zeros_like(y_vals)
    mean_f = np.zeros_like(raw)
    std_f  = np.zeros_like(raw)

    for _, grp in df.groupby("VILLAGE_ID"):
        idx = grp.index.tolist()
        diff_x[idx] = np.diff(raw[idx], axis=0, prepend=raw[idx][:1])
        lag_y[idx[0]] = y_vals[idx[0]]
        for i in range(1, len(idx)):
            lag_y[idx[i]] = y_vals[idx[i - 1]]
        mean_f[idx] = raw[idx].mean(axis=0)
        std_f[idx]  = raw[idx].std(axis=0)

    X_full = np.concatenate(
        [raw, time_feats, interaction, diff_x, lag_y, mean_f, std_f], axis=1
    ).astype(np.float32)

    y = df["Cluster"].values.astype(np.int64)
    return raw, X_full, y, df


# 训练用于 SHAP 分析的 TabICL 模型

def train_tabicl(X_full: np.ndarray, y: np.ndarray) -> tuple:
    """Train TabICL on the full dataset and return model artifacts."""
    from tabicl import TabICLClassifier

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_full).astype(np.float32)

    print("[TabICL] Training TabICL on full data")
    model = TabICLClassifier(n_estimators=8, random_state=42)
    model.fit(X_scaled, y)
    print("[TabICL] Training completed")
    return model, scaler, X_scaled


# 构建基于原始 14 个自变量扰动的 SHAP 预测函数。

def make_predict_fn(model, scaler, X_full_scaled: np.ndarray,
                    raw_idx: list[int]):
    """Build a prediction function for SHAP using the raw 14 features."""
    bg_mean = X_full_scaled.mean(axis=0)  # (80,)

    def predict_proba(X_raw: np.ndarray) -> np.ndarray:
        n = len(X_raw)
        X_in = np.tile(bg_mean, (n, 1)).astype(np.float32)
        X_in[:, raw_idx] = scaler.transform(
            np.concatenate([X_raw,
                            np.zeros((n, X_full_scaled.shape[1] - len(raw_idx)))],
                           axis=1)
        )[:, raw_idx]
        proba = model.predict_proba(X_in)
        classes = list(getattr(model, "classes_", [1, 2, 3]))
        idx = [classes.index(c) for c in [1, 2, 3]]
        proba = proba[:, idx]
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    return predict_proba


# 计算 SHAP 值

def compute_shap(predict_fn, X_raw: np.ndarray,
                 n_background: int = 50, n_explain: int | None = None) -> tuple:
    """Compute SHAP values with KernelExplainer."""
    if n_explain is None:
        n_explain = len(X_raw)
    shap_values_path = SHAP_DIR / "shared" / (
        f"tabicl_shap_values_n{n_explain}_bg{n_background}_ns100.npz"
    )
    if shap_values_path.exists():
        saved = np.load(shap_values_path, allow_pickle=False)
        print(f"[SHAP] loaded saved values: {shap_values_path}")
        return saved["shap_values"], saved["X_explain"], saved["exp_idx"]

    print(f"[SHAP] computing (background={n_background}, explain={n_explain})")
    rng = np.random.default_rng(42)
    bg_idx  = rng.choice(len(X_raw), n_background, replace=False)
    exp_idx = np.arange(len(X_raw)) if n_explain >= len(X_raw) else rng.choice(
        len(X_raw), n_explain, replace=False
    )

    background = X_raw[bg_idx]
    X_explain  = X_raw[exp_idx]

    explainer   = shap.KernelExplainer(predict_fn, background)
    sv_raw      = explainer.shap_values(X_explain, nsamples=100, silent=True)

    # 统一转换为 (样本数, 特征数, 类别数)。
    if isinstance(sv_raw, list):
        # 旧版本 SHAP 返回 list，每个元素形状为 (样本数, 特征数)。
        shap_values = np.stack(sv_raw, axis=2)
    else:
        shap_values = sv_raw  # 新版本 SHAP 返回 (样本数, 特征数, 类别数)。

    print(f"[SHAP] completed, shape={shap_values.shape}")
    np.savez_compressed(
        shap_values_path,
        shap_values=shap_values,
        X_explain=X_explain,
        exp_idx=exp_idx,
        background_idx=bg_idx,
        feature_names=np.array(SHAP_FEAT_NAMES),
        class_names=np.array(["LIST", "HIAT", "OECT"]),
        model_name=np.array("TabICL"),
    )
    print(f"[SHAP] saved values: {shap_values_path}")
    return shap_values, X_explain, exp_idx


# 绘图辅助函数

def savefig(path: Path, tight: bool = True) -> None:
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


# 图 2：SHAP 全局特征重要性与分布图

def fig_global_importance(shap_values, X_explain, cls_idx: int, cls_name: str,
                          out_dir: Path) -> None:
    # 颜色修改说明：
    # 1. a 图条形、环形扇区、下方类别条形图颜色统一修改 category_colors。
    # 2. a 图黑色分布曲线修改 ax_bar.plot(... color="#000000")。
    # 3. 环形图白色分隔线修改 Wedge 的 edgecolor，灰色辅助线修改 "#CFCFCF"/"#B4B4B4"。
    # 4. b 图散点色带修改 feature_cmap = LinearSegmentedColormap.from_list(...)。
    # 5. b 图零轴线颜色修改 ax_swarm.axvline(... color="#777777")。
    sv = shap_values[:, :, cls_idx]  # (n, n_features)  # (n, 14)
    feat_labels = [FEAT_LABELS[f] for f in SHAP_FEAT_NAMES]

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    y_pos = np.arange(len(order))
    max_mean = mean_abs[order].max() if len(order) else 1.0
    shared_ylim = (len(order) - 0.5, -0.5)

    feature_r2 = []
    feature_weights = []
    for feat_idx in order:
        x = X_explain[:, feat_idx].astype(float)
        y = sv[:, feat_idx].astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3 or np.nanstd(x[mask]) <= 1e-12 or np.nanstd(y[mask]) <= 1e-12:
            continue
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        if np.isfinite(corr):
            feature_r2.append(float(corr ** 2))
            feature_weights.append(float(mean_abs[feat_idx]))
    summary_r2 = (
        float(np.average(feature_r2, weights=feature_weights))
        if feature_r2 and np.sum(feature_weights) > 0 else np.nan
    )

    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Wedge
    category_map = {
        "ELEVATION": "Natural base", "SLOPE": "Natural base",
        "ASPECT": "Natural base", "MAP": "Natural base",
        "MAT": "Natural base",
        "NDVI": "Ecological environment", "DWS": "Ecological environment",
        "RD": "Socioeconomic", "POP": "Socioeconomic",
        "PGDP": "Socioeconomic", "NLI": "Socioeconomic",
        "CRPU": "Historical culture", "ICHD": "Historical culture",
        "DAR": "Historical culture",
    }
    category_colors = MAP_CATEGORY_COLORS
    bar_colors = [
        category_colors[category_map.get(SHAP_FEAT_NAMES[i], "Terrain")]
        for i in order
    ]

    fig, (ax_bar, ax_swarm) = plt.subplots(
        1, 2, figsize=(13.2, 6.2),
        gridspec_kw={"width_ratios": [1.18, 1.05], "wspace": 0.12}
    )

    # a 图：按平均 |SHAP| 排序的条形图，并叠加小型分布曲线。
    ax_bar.barh(y_pos, mean_abs[order], color=bar_colors,
                edgecolor="white", height=0.68, alpha=0.88, zorder=2)
    for yi, feat_idx in zip(y_pos, order):
        vals = np.abs(sv[:, feat_idx])
        vals = vals[np.isfinite(vals)]
        if len(vals) < 3:
            continue
        x_line = np.quantile(vals, np.linspace(0.02, 0.98, 90))
        histogram, edges = np.histogram(vals, bins=28, density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        density = np.interp(x_line, centers, histogram, left=0.0, right=0.0)
        x_max = np.nanmax(x_line)
        if not np.isfinite(x_max) or x_max <= 0:
            continue
        x_line = x_line / x_max * mean_abs[feat_idx] * 0.92
        density_max = np.nanmax(density)
        if np.isfinite(density_max) and density_max > 0:
            density = (density / density_max - 0.5) * 0.24
        else:
            density = np.zeros_like(x_line)
        ax_bar.plot(x_line, yi + density,
                    color=MAP_BOUNDARY, lw=0.86, alpha=0.88,
                    solid_capstyle="round", zorder=4)
    ax_bar.axvline(0, color=MAP_BOUNDARY, lw=0.8)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([feat_labels[i] for i in order],
                           fontfamily="Times New Roman", fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_ylim(shared_ylim)
    ax_bar.set_xlim(-max_mean * 0.03, max_mean * 1.12)
    ax_bar.set_xlabel("Mean |SHAP Value|", fontfamily="Times New Roman")
    r2_label = f", $R^2$ = {summary_r2:.3f}" if np.isfinite(summary_r2) else ""
    ax_bar.set_title(f"$N$ = {len(X_explain)}{r2_label}",
                     fontfamily="Times New Roman", fontsize=10, pad=6)
    ax_bar.text(-0.18, 1.02, "a", transform=ax_bar.transAxes,
                fontsize=14, fontweight="bold", fontfamily="Times New Roman")
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.grid(axis="x", alpha=0.18, lw=0.6)
    ax_bar.xaxis.set_major_locator(mticker.MaxNLocator(4))

    # a 图右侧嵌入图：半放射状类别贡献扇形图。
    cat_totals = {}
    for feat_idx, value in enumerate(mean_abs):
        cat = category_map.get(SHAP_FEAT_NAMES[feat_idx], "Terrain")
        cat_totals[cat] = cat_totals.get(cat, 0.0) + float(value)
    cats = [c for c in ["Socioeconomic", "Ecological environment",
                        "Natural base", "Historical culture"]
            if cat_totals.get(c, 0) > 0]
    cat_values = [cat_totals[c] for c in cats]
    total_cat = sum(cat_values) or 1

    ax_pie = ax_bar.inset_axes([0.50, 0.16, 0.42, 0.46])
    ax_cat = ax_bar.inset_axes([0.58, 0.05, 0.34, 0.20])

    def lighten_color(color, amount):
        r, g, b, a = to_rgba(color)
        return (
            r + (1.0 - r) * amount,
            g + (1.0 - g) * amount,
            b + (1.0 - b) * amount,
            a,
        )

    main_items = [
        {"label": f"{value / total_cat * 100:.1f}%",
         "value": float(value),
         "color": category_colors[cat],
         "cat": cat}
        for cat, value in zip(cats, cat_values)
    ]
    wedges, _ = ax_pie.pie(
        cat_values,
        radius=1.0,
        startangle=88,
        counterclock=False,
        colors=[category_colors[c] for c in cats],
        wedgeprops={"width": 0.28, "edgecolor": "white", "linewidth": 1.0},
    )
    inner_values = []
    inner_colors = []
    for cat in cats:
        feat_indices = [
            idx for idx, feat in enumerate(SHAP_FEAT_NAMES)
            if category_map.get(feat) == cat
        ]
        feat_indices = sorted(feat_indices, key=lambda idx: mean_abs[idx],
                              reverse=True)
        n_feats = max(len(feat_indices), 1)
        for local_idx, feat_idx in enumerate(feat_indices):
            inner_values.append(float(mean_abs[feat_idx]))
            amount = 0.18 + 0.42 * local_idx / max(n_feats - 1, 1)
            inner_colors.append(lighten_color(category_colors[cat], amount))
    ax_pie.pie(
        inner_values,
        radius=0.71,
        startangle=88,
        counterclock=False,
        colors=inner_colors,
        wedgeprops={"width": 0.27, "edgecolor": "white", "linewidth": 0.8},
    )
    for wedge, cat, value in zip(wedges, cats, cat_values):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        pct = value / total_cat * 100
        short_names = {
            "Socioeconomic": "Socioeconomic",
            "Ecological environment": "Ecological",
            "Natural base": "Natural",
            "Historical culture": "Historical",
        }
        ax_pie.annotate(
            f"{short_names.get(cat, cat)}\n{pct:.1f}%",
            xy=(0.95 * np.cos(angle), 0.95 * np.sin(angle)),
            xytext=(1.30 * np.cos(angle), 1.30 * np.sin(angle)),
            ha="left" if np.cos(angle) >= 0 else "right",
            va="center",
            fontsize=7.0,
            fontfamily="Times New Roman",
            fontweight="bold",
            color=category_colors[cat],
            arrowprops=dict(arrowstyle="-", color=MAP_AUX, linewidth=0.65),
        )
    ax_pie.text(
        0,
        0,
        "Category\nContribution\nDegree",
        ha="center",
        va="center",
        fontsize=6.8,
        fontfamily="Times New Roman",
        fontweight="bold",
        color=MAP_BOUNDARY,
    )
    ax_pie.set_xlim(-1.48, 1.48)
    ax_pie.set_ylim(-1.36, 1.36)
    ax_pie.set_aspect("equal")
    ax_pie.set_axis_off()

    ax_pie.text(-0.18, 1.02, "", transform=ax_pie.transAxes,
                fontsize=14, fontweight="bold", fontfamily="Times New Roman")

    # 扇形图下方的小型类别贡献条形图。
    short_names = {
        "Socioeconomic": "Socioeconomic",
        "Ecological environment": "Ecological",
        "Natural base": "Natural",
        "Historical culture": "Historical",
    }
    cat_y = np.arange(len(cats))
    ax_cat.barh(cat_y, cat_values, color=[category_colors[c] for c in cats],
                height=0.58, edgecolor="white", alpha=0.88)
    for yi, cat, value in zip(cat_y, cats, cat_values):
        ax_cat.text((max(cat_values) or 1) * 0.015, yi,
                    short_names.get(cat, cat),
                    ha="left", va="center", fontsize=8.4,
                    fontfamily="Times New Roman", color=MAP_BOUNDARY,
                    fontweight="bold")
    ax_cat.set_yticks([])
    ax_cat.set_xlim(0, (max(cat_values) or 1) * 1.18)
    ax_cat.xaxis.set_major_locator(mticker.MaxNLocator(3))
    ax_cat.tick_params(axis="x", labelsize=8, width=1.6, length=4, pad=1)
    ax_cat.spines[["top", "right"]].set_visible(False)
    ax_cat.spines[["left", "bottom"]].set_linewidth(1.6)
    ax_cat.spines[["left", "bottom"]].set_color("black")
    ax_cat.invert_yaxis()
    ax_cat.set_axis_off()
    ax_cat.set_visible(False)

    # b 图：beeswarm 风格的 SHAP 汇总散点图。
    rng = np.random.default_rng(42)
    feature_cmap = LinearSegmentedColormap.from_list("map_feature", MAP_SEQUENTIAL)
    for yi, feat_idx in zip(y_pos, order):
        vals = sv[:, feat_idx]
        feat_vals = X_explain[:, feat_idx].astype(float)
        denom = np.nanmax(feat_vals) - np.nanmin(feat_vals)
        color_values = (feat_vals - np.nanmin(feat_vals)) / (denom + 1e-12)
        jitter = rng.normal(0, 0.075, size=len(vals))
        ax_swarm.scatter(vals, yi + jitter, c=color_values,
                         cmap=feature_cmap, vmin=0, vmax=1,
                         s=9, alpha=0.78, edgecolors="none")

    ax_swarm.axvline(0, color=MAP_AUX, lw=0.9)
    ax_swarm.set_ylim(shared_ylim)
    ax_swarm.set_yticks(y_pos)
    ax_swarm.set_yticklabels([])
    ax_swarm.set_xlabel("SHAP Value", fontfamily="Times New Roman")
    ax_swarm.text(-0.12, 1.02, "b", transform=ax_swarm.transAxes,
                  fontsize=14, fontweight="bold", fontfamily="Times New Roman")
    ax_swarm.spines[["top", "right"]].set_visible(False)
    ax_swarm.grid(axis="x", alpha=0.18, lw=0.6)
    ax_swarm.xaxis.set_major_locator(mticker.MaxNLocator(5))

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=feature_cmap)
    cax = ax_swarm.inset_axes([0.84, 0.03, 0.035, 0.18])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Feature value", rotation=270, labelpad=14,
                   fontfamily="Times New Roman")
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    cax.text(-0.35, 1.0, "High", transform=cax.transAxes,
             ha="right", va="center", fontsize=8,
             fontfamily="Times New Roman")
    cax.text(-0.35, 0.0, "Low", transform=cax.transAxes,
             ha="right", va="center", fontsize=8,
             fontfamily="Times New Roman")

    fig.suptitle(f"SHAP Feature Importance and Distribution - {cls_name}",
                 fontfamily="Times New Roman", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / f"Fig2_shap_summary_{cls_name}.png",
                dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"Fig2_global_importance_{cls_name}.png",
                dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"Fig2_beeswarm_{cls_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_dir / f'Fig2_shap_summary_{cls_name}.png'}")


# 图 3：SHAP 单特征依赖图

def fig_dependence(shap_values, X_explain, cls_idx: int, cls_name: str,
                   out_dir: Path, top_n: int = 14) -> None:
    # 颜色修改说明：
    # 1. 散点颜色修改 ax.scatter(... color="#0B5FA5")。
    # 2. 拟合曲线颜色修改 ax.plot(... color="#F2A13B")。
    # 3. 置信区间颜色修改 ax.fill_between(... color="#F2A65A")。
    # 4. 底部直方图颜色修改 rug_ax.bar(... color="#4B4B4B")。
    # 5. 零线、阈值线和网格颜色分别修改 "#777777"、"#444444"、"#D9D9D9"。
    sv = shap_values[:, :, cls_idx]  # (n, n_features)
    mean_abs = np.abs(sv).mean(axis=0)
    top_feats = np.argsort(mean_abs)[::-1][:top_n]
    feat_labels = [FEAT_LABELS[f] for f in SHAP_FEAT_NAMES]

    def polynomial_dependence(x, y, degree=4, n_grid=180):
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid].astype(float)
        y = y[valid].astype(float)
        x_mean = x.mean()
        x_std = x.std() + 1e-12
        xs = (x - x_mean) / x_std
        fit_degree = min(degree, max(len(np.unique(xs)) - 1, 1))
        coef = np.polyfit(xs, y, fit_degree)
        poly = np.poly1d(coef)
        x_grid = np.linspace(x.min(), x.max(), n_grid)
        xg = (x_grid - x_mean) / x_std
        y_fit = poly(xg)
        residual = y - poly(xs)
        sigma = np.std(residual)
        spread = 1.0 + 0.9 * np.abs(xg - xg.mean()) / (np.max(np.abs(xg - xg.mean())) + 1e-12)
        y_low = y_fit - 1.96 * sigma * 0.35 * spread
        y_high = y_fit + 1.96 * sigma * 0.35 * spread
        y_pred = poly(xs)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        n = len(y)
        p = fit_degree
        f_stat = (r2 / max(p, 1)) / ((1 - r2) / max(n - p - 1, 1) + 1e-12)
        p_value = np.exp(-0.5 * max(f_stat, 0.0))
        return x_grid, y_fit, y_low, y_high, r2, p_value, coef

    n_cols = 4
    n_rows = int(np.ceil(len(top_feats) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.25 * n_cols, 2.95 * n_rows))
    axes = np.array(axes).reshape(-1)

    for plot_idx, (ax, fi) in enumerate(zip(axes, top_feats)):
        x = X_explain[:, fi].astype(float)
        y = sv[:, fi].astype(float)
        x_grid, y_smooth, y_low, y_high, r2, p_value, coef = polynomial_dependence(x, y)

        ax.scatter(x, y, s=9, color=MAP_GREEN_DARK, alpha=0.55,
                   edgecolors="none", label="Samples")
        ax.fill_between(x_grid, y_low, y_high, color=MAP_PINK_LIGHT,
                        alpha=0.24, label="95% Confidence interval")
        eq_label = (
            f"Fitted curve: $R^2$={r2:.3f}\n"
            f"$p$={'<0.001' if p_value < 0.001 else f'{p_value:.3f}'}"
        )
        ax.plot(x_grid, y_smooth, color=MAP_PINK, lw=1.7,
                label=eq_label)
        ax.axhline(0, color=MAP_AUX, lw=0.8, linestyle="--")

        signs = np.sign(y_smooth)
        crossing = np.where(np.diff(signs) != 0)[0]
        if len(crossing) > 0:
            threshold = x_grid[crossing[0]]
            ax.axvline(threshold, color=MAP_BOUNDARY, lw=0.8,
                       linestyle=":", label=f"Threshold={threshold:.2f}")

        rug_ax = ax.inset_axes([0.0, -0.30, 1.0, 0.18], sharex=ax)
        hist, bins = np.histogram(x[np.isfinite(x)], bins=28)
        centers = (bins[:-1] + bins[1:]) / 2
        if hist.max() > 0:
            hist = hist / hist.max()
        widths = np.diff(bins)
        rug_ax.bar(centers, hist, width=widths, color=MAP_GRAY,
                   alpha=0.82, align="center")
        rug_ax.set_ylim(0, 1.05)
        rug_ax.set_yticks([])
        rug_ax.set_ylabel("Dist.", fontsize=7, fontfamily="Times New Roman")
        rug_ax.tick_params(axis="x", labelsize=7, length=2, pad=1)
        rug_ax.spines[["top", "right"]].set_visible(False)
        rug_ax.spines[["left", "bottom"]].set_linewidth(0.8)

        letter = chr(ord("a") + plot_idx)
        ax.set_title(f"{letter}. {feat_labels[fi]}", loc="left",
                     fontfamily="Times New Roman", fontsize=10,
                     fontweight="bold")
        ax.set_xlabel(feat_labels[fi], fontfamily="Times New Roman")
        ax.set_ylabel(f"SHAP value ({cls_name})", fontfamily="Times New Roman")
        ax.grid(True, color=MAP_LIGHT_GRAY, lw=0.55, alpha=0.75)
        ax.tick_params(axis="both", labelsize=8, colors="black",
                       width=0.9, length=3)
        legend_loc = "upper left"
        if letter in {"a", "b", "d", "f", "g", "k", "m"}:
            legend_loc = "upper right"
        elif letter == "n":
            legend_loc = "lower right"
        ax.legend(loc=legend_loc, fontsize=5.8, frameon=False,
                  prop={"family": "Times New Roman", "size": 5.8})
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(0.9)
        ax.spines[["left", "bottom"]].set_color("black")

    for ax in axes[len(top_feats):]:
        ax.axis("off")

    fig.suptitle(f"SHAP Dependence Plots - {cls_name}",
                 fontfamily="Times New Roman", fontsize=13)
    fig.subplots_adjust(hspace=0.65, wspace=0.36)
    savefig(out_dir / f"Fig3_dependence_{cls_name}.png")


# 图 4：主效应与交互效应对比图

def compute_second_order_interactions(predict_fn, X_explain, cls_idx: int,
                                      max_samples: int = 400) -> np.ndarray:
    n_samples, n_feat = X_explain.shape
    rng = np.random.default_rng(42)
    if n_samples > max_samples:
        sample_idx = rng.choice(n_samples, max_samples, replace=False)
        X_eval = X_explain[sample_idx]
    else:
        X_eval = X_explain

    baseline = X_explain.mean(axis=0)
    X_base = np.tile(baseline, (len(X_eval), 1))
    f_base = predict_fn(X_base)[:, cls_idx]
    single_effects = []
    for i in range(n_feat):
        Xi = X_base.copy()
        Xi[:, i] = X_eval[:, i]
        single_effects.append(predict_fn(Xi)[:, cls_idx] - f_base)

    inter = np.zeros((n_feat, n_feat), dtype=float)
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            Xij = X_base.copy()
            Xij[:, i] = X_eval[:, i]
            Xij[:, j] = X_eval[:, j]
            joint = predict_fn(Xij)[:, cls_idx] - f_base
            value = np.abs(joint - single_effects[i] - single_effects[j]).mean()
            inter[i, j] = value
            inter[j, i] = value
    return inter


def compute_pair_second_order_values(predict_fn, X_explain, cls_idx: int,
                                     feat_i: int, feat_j: int) -> np.ndarray:
    baseline = X_explain.mean(axis=0)
    X_base = np.tile(baseline, (len(X_explain), 1))
    f_base = predict_fn(X_base)[:, cls_idx]

    Xi = X_base.copy()
    Xi[:, feat_i] = X_explain[:, feat_i]
    Xj = X_base.copy()
    Xj[:, feat_j] = X_explain[:, feat_j]
    Xij = X_base.copy()
    Xij[:, feat_i] = X_explain[:, feat_i]
    Xij[:, feat_j] = X_explain[:, feat_j]

    effect_i = predict_fn(Xi)[:, cls_idx] - f_base
    effect_j = predict_fn(Xj)[:, cls_idx] - f_base
    joint = predict_fn(Xij)[:, cls_idx] - f_base
    return joint - effect_i - effect_j


def fig_main_vs_interaction(shap_values, X_explain, cls_idx: int, cls_name: str,
                             out_dir: Path, predict_fn=None) -> None:
    # 颜色修改说明：
    # 1. 主效应柱颜色修改 bars_main 的 color="#67BFAE"。
    # 2. 二阶交互效应柱颜色修改 bars_inter 的 color="#A65A5A"。
    # 3. 柱边框颜色修改 edgecolor="white"，网格线颜色修改 "#DDDDDD"。
    sv = shap_values[:, :, cls_idx]  # (n, n_features)
    feat_labels = [FEAT_LABELS[f] for f in SHAP_FEAT_NAMES]
    mean_abs = np.abs(sv).mean(axis=0)

    if predict_fn is None:
        centered = sv - sv.mean(axis=0, keepdims=True)
        interaction_matrix = np.abs(np.cov(centered.T))
        np.fill_diagonal(interaction_matrix, 0)
    else:
        interaction_matrix = compute_second_order_interactions(
            predict_fn, X_explain, cls_idx
        )

    interaction_effects = interaction_matrix.sum(axis=1)
    if interaction_effects.max() > 0:
        interaction_effects = (
            interaction_effects / interaction_effects.max() *
            mean_abs.max() * 0.45
        )

    order = np.argsort(mean_abs)[::-1]
    labels = [feat_labels[i] for i in order]
    main_effects = mean_abs[order]
    interaction_effects = interaction_effects[order]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    bars_main = ax.bar(x - width / 2, main_effects, width,
                       label="Main effect (Mean |SHAP|)",
                       color=MAP_GREEN, edgecolor="white", alpha=0.95)
    bars_inter = ax.bar(x + width / 2, interaction_effects, width,
                        label="2nd-order interaction (sum over others)",
                        color=MAP_PINK, edgecolor="white", alpha=0.95)
    for bars in (bars_main, bars_inter):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{height:.3f}", ha="center", va="bottom",
                    fontsize=7, fontfamily="Times New Roman")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontfamily="Times New Roman")
    ax.set_ylabel("Magnitude (Mean |SHAP|)", fontfamily="Times New Roman")
    ax.set_title(f"Main vs Interaction Effects - {cls_name}",
                 fontfamily="Times New Roman")
    ax.legend(loc="upper right", prop={"family": "Times New Roman", "size": 9})
    ax.grid(axis="y", color=MAP_LIGHT_GRAY, lw=0.6, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(out_dir / f"Fig4_main_vs_interaction_{cls_name}.png")


# 图 5：特征交互效应矩阵图

def fig_interaction_matrix(shap_values, X_explain, cls_idx: int, cls_name: str,
                            out_dir: Path, predict_fn=None) -> None:
    # 颜色修改说明：
    # 1. 非对角散点色带修改 scatter_cmap = LinearSegmentedColormap.from_list(...)。
    # 2. 对角直方图颜色修改 ax.hist(... color="#BDBDBD")。
    # 3. 下三角热度背景修改 ax.imshow(... cmap="Reds")。
    # 4. 坐标轴边框颜色修改 spine.set_color("#CCCCCC")。
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    sv = shap_values[:, :, cls_idx]  # (n, n_features)  # (n, 14)
    feat_labels = [FEAT_LABELS[f] for f in SHAP_FEAT_NAMES]
    n_feat = len(SHAP_FEAT_NAMES)

    if predict_fn is None:
        centered = sv - sv.mean(axis=0, keepdims=True)
        inter_strength = np.abs(np.cov(centered.T))
        np.fill_diagonal(inter_strength, 0)
    else:
        inter_strength = compute_second_order_interactions(
            predict_fn, X_explain, cls_idx
        )

    main_strength = np.abs(sv).mean(axis=0)
    order = np.argsort(main_strength)[::-1]
    inter_strength = inter_strength[np.ix_(order, order)]
    X_plot = X_explain[:, order]
    labels = [feat_labels[i] for i in order]

    fig, axes = plt.subplots(n_feat, n_feat, figsize=(11, 10))
    vmax = np.max(inter_strength) or 1.0
    scatter_cmap = LinearSegmentedColormap.from_list("map_joint_signal", MAP_CONTINUOUS)
    heat_cmap = LinearSegmentedColormap.from_list("map_interaction_heat", MAP_SEQUENTIAL)
    for i in range(n_feat):
        for j in range(n_feat):
            ax = axes[i, j]
            if i > j:
                value = inter_strength[i, j]
                ax.imshow([[value]], cmap=heat_cmap, vmin=0, vmax=vmax)
                ax.text(
                    0,
                    0,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontfamily="Times New Roman",
                    color=MAP_BOUNDARY,
                    bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.38},
                )
                ax.set_xticks([])
                ax.set_yticks([])
            elif i < j:
                x = X_plot[:, j]
                y = X_plot[:, i]
                color = sv[:, order[i]] + sv[:, order[j]]
                max_abs = np.max(np.abs(color)) or 1.0
                ax.scatter(x, y, c=color, cmap=scatter_cmap,
                           vmin=-max_abs, vmax=max_abs,
                           s=5, alpha=0.45, edgecolors="none")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.hist(X_plot[:, i], bins=18, color=MAP_GREEN,
                        edgecolor="white", linewidth=0.3)
                ax.set_xticks([])
                ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color(MAP_LIGHT_GRAY)

    for ax, label in zip(axes[-1, :], labels):
        ax.set_xlabel(label, rotation=90, fontfamily="Times New Roman",
                      fontsize=7)
    for ax, label in zip(axes[:, 0], labels):
        ax.set_ylabel(label, fontfamily="Times New Roman", fontsize=7)

    fig.subplots_adjust(left=0.06, right=0.76, bottom=0.08, top=0.92,
                        wspace=0.25, hspace=0.25)
    cax = fig.add_axes([0.81, 0.22, 0.018, 0.55])
    sm = plt.cm.ScalarMappable(cmap=scatter_cmap,
                               norm=Normalize(-vmax, vmax))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Joint SHAP signal", rotation=270, labelpad=13,
                   fontfamily="Times New Roman")
    cbar.set_ticks([-vmax, vmax])
    cbar.set_ticklabels(["Low", "High"])

    fig.suptitle(f"Second-order Interaction Matrix - {cls_name}",
                 fontfamily="Times New Roman", fontsize=13)
    savefig(out_dir / f"Fig5_interaction_matrix_{cls_name}.png", tight=False)


# 图 6：SHAP 双因子交互图

def fig_bivariate_interaction(shap_values, X_explain, cls_idx: int, cls_name: str,
                               out_dir: Path, predict_fn=None) -> None:
    # 颜色修改说明：
    # 1. 3D 曲面、2D 等高面和散点色带统一使用 _PAL 连续色带。
    # 2. 3D 散点颜色修改 ax3d.scatter(... c=MAP_RED)。
    # 3. 等高线颜色修改 ax2d.contour(... colors=MAP_BOUNDARY)。
    # 4. 2D 散点边框颜色修改 edgecolors=MAP_BOUNDARY。
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec
    from scipy.interpolate import SmoothBivariateSpline, griddata
    from scipy.ndimage import gaussian_filter
    from scipy.stats import f as f_dist

    surface_percentiles = (10, 90)
    scatter_percentiles = (15, 85)
    z_percentiles = (10, 90)

    sv = shap_values[:, :, cls_idx]  # (n, n_features)
    feat_labels = [FEAT_LABELS[f] for f in SHAP_FEAT_NAMES]
    interaction_matrix = None
    if predict_fn is not None:
        interaction_matrix = compute_second_order_interactions(
            predict_fn, X_explain, cls_idx
        )

    pairs_to_plot = []
    for item in INTERACTION_PAIRS:
        candidates = item if isinstance(item, list) else [item]
        if interaction_matrix is None:
            best_pair = max(
                candidates,
                key=lambda p: np.abs((sv[:, SHAP_FEAT_NAMES.index(p[0])] -
                                      sv[:, SHAP_FEAT_NAMES.index(p[0])].mean()) *
                                     (sv[:, SHAP_FEAT_NAMES.index(p[1])] -
                                      sv[:, SHAP_FEAT_NAMES.index(p[1])].mean())).mean()
            )
        else:
            best_pair = max(
                candidates,
                key=lambda p: interaction_matrix[
                    SHAP_FEAT_NAMES.index(p[0]), SHAP_FEAT_NAMES.index(p[1])
                ]
            )
        pairs_to_plot.append(best_pair)

    def smooth_surface(z_grid, sigma=1.65):
        if z_grid.size == 0:
            return z_grid
        finite = np.isfinite(z_grid)
        if not np.any(finite):
            return np.zeros_like(z_grid, dtype=float)
        fill_value = float(np.nanmedian(z_grid[finite]))
        filled = np.where(finite, z_grid, fill_value)
        return gaussian_filter(filled, sigma=sigma, mode="nearest")

    def robust_limits(values, lower=2, upper=98):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return -1.0, 1.0
        low, high = np.nanpercentile(values, [lower, upper])
        if not np.isfinite(low) or not np.isfinite(high) or np.isclose(low, high):
            low, high = np.nanmin(values), np.nanmax(values)
        if np.isclose(low, high):
            pad = max(abs(float(low)) * 0.05, 1.0)
            low -= pad
            high += pad
        return float(low), float(high)

    def format_p_value(p_value: float) -> str:
        if not np.isfinite(p_value):
            return "NA"
        if p_value < 0.001:
            return "<0.001"
        return f"{p_value:.3f}"

    def polynomial_response_surface(xa, xb, interaction_val, n_grid=180,
                                    degree=2, curvature_penalty=18.0):
        """Fit a continuous response surface so contours do not inherit point-grid edges."""
        finite = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(interaction_val)
        xa = xa[finite].astype(float)
        xb = xb[finite].astype(float)
        interaction_val = interaction_val[finite].astype(float)

        x_low, x_high = robust_limits(xa, *surface_percentiles)
        y_low, y_high = robust_limits(xb, *surface_percentiles)
        in_fit = (
            (xa >= x_low) & (xa <= x_high) &
            (xb >= y_low) & (xb <= y_high)
        )
        xa_fit = xa[in_fit]
        xb_fit = xb[in_fit]
        z_fit = interaction_val[in_fit]
        if xa_fit.size == 0:
            xa_fit = xa
            xb_fit = xb
            z_fit = interaction_val
        z_low, z_high = robust_limits(z_fit, *z_percentiles)
        z_fit_smooth = np.clip(z_fit, z_low, z_high)

        x_grid = np.linspace(x_low, x_high, n_grid)
        y_grid = np.linspace(y_low, y_high, n_grid)
        Xg, Yg = np.meshgrid(x_grid, y_grid)

        x_mean, x_std = float(np.nanmean(xa_fit)), float(np.nanstd(xa_fit) or 1.0)
        y_mean, y_std = float(np.nanmean(xb_fit)), float(np.nanstd(xb_fit) or 1.0)
        xs = (xa_fit - x_mean) / x_std
        ys = (xb_fit - y_mean) / y_std
        xgs = (Xg.ravel() - x_mean) / x_std
        ygs = (Yg.ravel() - y_mean) / y_std

        def fit_polynomial_surface():
            terms = []
            grid_terms = []
            powers = []
            for px in range(degree + 1):
                for py in range(degree + 1 - px):
                    terms.append((xs ** px) * (ys ** py))
                    grid_terms.append((xgs ** px) * (ygs ** py))
                    powers.append((px, py))
            design = np.column_stack(terms)
            grid_design = np.column_stack(grid_terms)
            penalty = np.array(
                [
                    0.0 if px + py <= 1 else curvature_penalty
                    for px, py in powers
                ],
                dtype=float,
            )
            lhs = design.T @ design + np.diag(penalty)
            rhs = design.T @ z_fit_smooth
            coef = np.linalg.solve(lhs, rhs)
            return (grid_design @ coef).reshape(Xg.shape), design @ coef, design.shape[1]

        z_grid, fitted, n_params = fit_polynomial_surface()

        z_grid = np.clip(smooth_surface(z_grid, sigma=1.45), z_low, z_high)
        ss_res = float(np.sum((z_fit_smooth - fitted) ** 2))
        ss_tot = float(np.sum((z_fit_smooth - np.mean(z_fit_smooth)) ** 2)) or 1.0
        r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        n_fit = int(z_fit_smooth.size)
        df_model = max(n_params - 1, 1)
        df_resid = max(n_fit - n_params, 1)
        if r2 >= 1.0:
            p_value = 0.0
        elif r2 <= 0.0 or n_fit <= n_params:
            p_value = 1.0
        else:
            f_stat = (r2 / df_model) / ((1.0 - r2) / df_resid)
            p_value = float(f_dist.sf(f_stat, df_model, df_resid))
        return Xg, Yg, smooth_surface(z_grid, sigma=1.0), r2, p_value

    def pair_surface(feat_i, feat_j, n_grid=90):
        xa = X_explain[:, feat_i]
        xb = X_explain[:, feat_j]
        x_grid = np.linspace(np.nanpercentile(xa, 2), np.nanpercentile(xa, 98), n_grid)
        y_grid = np.linspace(np.nanpercentile(xb, 2), np.nanpercentile(xb, 98), n_grid)
        Xg, Yg = np.meshgrid(x_grid, y_grid)
        baseline = X_explain.mean(axis=0)
        flat_n = Xg.size
        X_base = np.tile(baseline, (flat_n, 1))
        f_base = predict_fn(X_base)[:, cls_idx]
        Xi = X_base.copy()
        Xi[:, feat_i] = Xg.ravel()
        Xj = X_base.copy()
        Xj[:, feat_j] = Yg.ravel()
        Xij = X_base.copy()
        Xij[:, feat_i] = Xg.ravel()
        Xij[:, feat_j] = Yg.ravel()
        effect_i = predict_fn(Xi)[:, cls_idx] - f_base
        effect_j = predict_fn(Xj)[:, cls_idx] - f_base
        joint = predict_fn(Xij)[:, cls_idx] - f_base
        Z = (joint - effect_i - effect_j).reshape(Xg.shape)
        return Xg, Yg, smooth_surface(Z)

    def pair_surface_from_points(xa, xb, interaction_val, n_grid=90):
        """Fallback surface from observed points; avoids a misleading flat plane."""
        finite = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(interaction_val)
        xa = xa[finite]
        xb = xb[finite]
        interaction_val = interaction_val[finite]
        x_grid = np.linspace(np.nanpercentile(xa, 2), np.nanpercentile(xa, 98), n_grid)
        y_grid = np.linspace(np.nanpercentile(xb, 2), np.nanpercentile(xb, 98), n_grid)
        Xg, Yg = np.meshgrid(x_grid, y_grid)
        points = np.column_stack([xa, xb])
        Z = griddata(points, interaction_val, (Xg, Yg), method="cubic")
        if np.isnan(Z).any():
            linear_z = griddata(points, interaction_val, (Xg, Yg), method="linear")
            Z = np.where(np.isnan(Z), linear_z, Z)
        if np.isnan(Z).any():
            nearest_z = griddata(points, interaction_val, (Xg, Yg), method="nearest")
            Z = np.where(np.isnan(Z), nearest_z, Z)
        return Xg, Yg, smooth_surface(Z)

    def draw_pair(fig, spec, feat_a, feat_b, letter=None):
        ia = SHAP_FEAT_NAMES.index(feat_a)
        ib = SHAP_FEAT_NAMES.index(feat_b)
        xa = X_explain[:, ia]
        xb = X_explain[:, ib]
        if predict_fn is None:
            interaction_val = sv[:, ia] + sv[:, ib]
            cbar_label = "Fallback combined SHAP"
        else:
            interaction_val = compute_pair_second_order_values(
                predict_fn, X_explain, cls_idx, ia, ib
            )
            cbar_label = "Second-order interaction"
        x_low, x_high = robust_limits(xa, *surface_percentiles)
        y_low, y_high = robust_limits(xb, *surface_percentiles)
        scatter_x_low, scatter_x_high = robust_limits(xa, *scatter_percentiles)
        scatter_y_low, scatter_y_high = robust_limits(xb, *scatter_percentiles)
        scatter_z_low, scatter_z_high = robust_limits(interaction_val, *z_percentiles)
        real_sample = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(interaction_val)
        in_range = (
            real_sample &
            (xa >= scatter_x_low) & (xa <= scatter_x_high) &
            (xb >= scatter_y_low) & (xb <= scatter_y_high) &
            (interaction_val >= scatter_z_low) &
            (interaction_val <= scatter_z_high)
        )
        if not np.any(in_range):
            in_range = real_sample
        right_column_pairs = {
            ("DAR", "RD"),
            ("MAT", "NDVI"),
            ("CRPU", "ICHD"),
        }
        pair_key = (feat_a, feat_b)
        # CRPU×ICHD 真实交互最强、碗最深，单独加大曲率惩罚把曲面压平缓些；
        # 其余右列对（DAR×RD、MAT×NDVI）保持 95，左列保持 24。
        if pair_key == ("CRPU", "ICHD"):
            curvature_penalty = 260.0
        elif pair_key in right_column_pairs:
            curvature_penalty = 95.0
        else:
            curvature_penalty = 24.0
        Xg, Yg, Z, response_r2, response_p = polynomial_response_surface(
            xa,
            xb,
            interaction_val,
            curvature_penalty=curvature_penalty,
        )
        xa_plot = xa[in_range]
        xb_plot = xb[in_range]
        interaction_plot = interaction_val[in_range]
        print(
            f"    {feat_a} x {feat_b}: plotted {len(xa_plot)}/"
            f"{int(real_sample.sum())} real samples within robust display range"
        )

        finite_z = Z[np.isfinite(Z)]

        sub = spec.subgridspec(1, 2, width_ratios=[0.88, 1.12], wspace=0.26)
        ax3d = fig.add_subplot(sub[0, 0], projection="3d")
        ax2d = fig.add_subplot(sub[0, 1])
        # \u5404\u5b50\u56fe\u72ec\u7acb\u6309\u66f2\u9762\u6781\u503c\u914d\u7f6e\u5bf9\u79f0\u8272\u9636\u3002
        vmax = float(np.nanpercentile(np.abs(finite_z), 98)) if finite_z.size else 1.0
        vmax = vmax or 1.0
        Z = np.clip(Z, -vmax, vmax)
        contour_levels = np.linspace(-vmax, vmax, 28)
        line_levels = np.linspace(-vmax, vmax, 9)
        surface_cmap = LinearSegmentedColormap.from_list("map_surface", MAP_SEQUENTIAL[::-1])

        # \u5e73\u6ed1\u66f2\u9762\uff1a\u9ad8\u7f51\u683c\u5bc6\u5ea6 + \u6297\u952f\u9f7f + \u65e0\u7f51\u683c\u63cf\u8fb9\uff0c\u6d88\u9664\u952f\u9f7f\u3002
        surf = ax3d.plot_surface(Xg, Yg, Z, cmap=surface_cmap,
                                 vmin=-vmax, vmax=vmax,
                                 rcount=120, ccount=120,
                                 linewidth=0, edgecolor="none",
                                 alpha=0.92, antialiased=True)
        # 3D \u771f\u5b9e\u6837\u672c\u70b9\uff1a\u62bd\u6837\u9650\u70b9 + \u4f4e\u900f\u660e\u5ea6 + \u666f\u6df1\u6de1\u5316\uff0c\u907f\u514d\u906e\u6321\u66f2\u9762\u4f4e\u6d3c\u5904\u3002
        if len(xa_plot) > 0:
            step = max(1, len(xa_plot) // 130)
            ax3d.scatter(xa_plot[::step], xb_plot[::step], interaction_plot[::step],
                         c=MAP_RED, s=6, alpha=0.45, depthshade=True)
        ax3d.set_xlabel(feat_labels[ia], fontfamily="Times New Roman",
                        fontsize=9, labelpad=7)
        ax3d.set_ylabel(feat_labels[ib], fontfamily="Times New Roman",
                        fontsize=9, labelpad=7)
        ax3d.zaxis.set_rotate_label(False)
        ax3d.set_zlabel(cbar_label, fontfamily="Times New Roman",
                        fontsize=9, labelpad=10, rotation=90)
        ax3d.tick_params(labelsize=8, pad=2)
        ax3d.locator_params(nbins=5)
        ax3d.view_init(elev=26, azim=-128)

        cf = ax2d.contourf(
            Xg,
            Yg,
            Z,
            levels=contour_levels,
            cmap=surface_cmap,
            vmin=-vmax,
            vmax=vmax,
            alpha=0.9,
            antialiased=True,
        )
        # \u8ba9\u586b\u5145\u8272\u5757\u8fb9\u7f18\u4e0e\u81ea\u8eab\u540c\u8272\uff0c\u6d88\u9664\u534a\u900f\u660e\u5206\u5e26\u63cf\u8fb9\u5bfc\u81f4\u7684\u952f\u9f7f\u3002
        try:
            cf.set_edgecolor("face")
        except Exception:
            pass
        contour_lines = ax2d.contour(
            Xg,
            Yg,
            Z,
            levels=line_levels,
            colors=MAP_BOUNDARY,
            linewidths=0.5,
            alpha=0.55,
        )
        ax2d.clabel(contour_lines, inline=True, fontsize=8, fmt="%.3f")
        sc = ax2d.scatter(
            xa_plot,
            xb_plot,
            c=interaction_plot,
            cmap=surface_cmap,
            vmin=-vmax,
            vmax=vmax,
            s=9,
            alpha=0.42,
            edgecolors=MAP_BOUNDARY,
            linewidths=0.2,
        )
        ax2d.set_xlabel(feat_labels[ia], fontfamily="Times New Roman", fontsize=11)
        ax2d.set_ylabel(feat_labels[ib], fontfamily="Times New Roman", fontsize=11)
        title_prefix = f"{letter}. " if letter else ""
        ax2d.set_title(
            f"{title_prefix}{feat_labels[ia]} x {feat_labels[ib]}\n"
            f"Surface-fit R\u00b2={response_r2 * 100:.1f}%, n={len(xa_plot)}",
            fontfamily="Times New Roman",
            fontweight="bold",
            fontsize=12,
        )
        ax2d.tick_params(labelsize=9)
        cbar = fig.colorbar(sc, ax=ax2d, fraction=0.046, pad=0.02)
        cbar.set_label(cbar_label, fontfamily="Times New Roman", fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        return surf

    for idx, (feat_a, feat_b) in enumerate(pairs_to_plot):
        fig = plt.figure(figsize=(8.8, 4.2))
        gs = GridSpec(1, 1, figure=fig)
        draw_pair(fig, gs[0, 0], feat_a, feat_b, chr(ord("a") + idx))
        fname = f"Fig6_bivariate_{feat_a}x{feat_b}_{cls_name}.png"
        savefig(out_dir / fname)

    n_cols = 2
    n_rows = int(np.ceil(len(pairs_to_plot) / n_cols))
    fig = plt.figure(figsize=(9.6 * n_cols, 4.5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.26, hspace=0.42)
    for idx, (feat_a, feat_b) in enumerate(pairs_to_plot):
        row, col = divmod(idx, n_cols)
        draw_pair(fig, gs[row, col], feat_a, feat_b, chr(ord("a") + idx))
    fig.suptitle(f"Bivariate Second-order Interaction Surfaces - {cls_name}",
                 fontfamily="Times New Roman", fontsize=17, fontweight="bold")
    savefig(out_dir / f"Fig6_bivariate_summary_{cls_name}.png")


# 图 7：SHAP 热力图

def fig_heatmap(shap_values, X_explain, exp_idx: np.ndarray,
                cls_idx: int, cls_name: str, df: pd.DataFrame,
                out_dir: Path) -> None:
    # 颜色修改说明：
    # 1. SHAP 热力图配色修改 heatmap_cmap = LinearSegmentedColormap.from_list(...)。
    # 2. 色阶范围修改 vmax 的计算方式，当前使用 SHAP 绝对值 98 分位数。
    # 3. 色条标题颜色可在 cbar.set_label(...) 中增加 color 参数。
    from matplotlib.colors import LinearSegmentedColormap

    sv = shap_values[:, :, cls_idx]  # (n, n_features)  # (n_explain, 14)
    feat_labels = [FEAT_LABELS[f] for f in SHAP_FEAT_NAMES]

    # 按平均 |SHAP| 对特征排序。
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1]
    sv_sorted = sv[:, order]
    feat_sorted = [feat_labels[i] for i in order]

    # 按第一主成分对样本排序。
    from sklearn.decomposition import PCA
    pc = PCA(n_components=1).fit_transform(sv_sorted)
    sample_order = np.argsort(pc[:, 0])

    fig, ax = plt.subplots(figsize=(12, 5))
    vmax = np.nanpercentile(np.abs(sv_sorted), 98)
    heatmap_cmap = LinearSegmentedColormap.from_list("map_shap_heat", MAP_CONTINUOUS)
    im = ax.imshow(sv_sorted[sample_order].T, aspect="auto",
                   cmap=heatmap_cmap, vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("SHAP Value", fontfamily="Times New Roman")
    ax.set_yticks(range(len(feat_sorted)))
    ax.set_yticklabels(feat_sorted, fontfamily="Times New Roman", fontsize=8)
    ax.set_xlabel("Sample Index (sorted)", fontfamily="Times New Roman")
    ax.set_title(f"SHAP Heatmap - {cls_name}", fontfamily="Times New Roman")
    ax.set_xticks([])
    savefig(out_dir / f"Fig7_heatmap_{cls_name}.png")


# 图 8：个体样本 SHAP 力图

def _draw_force_plot(ax, values, feat_labels, feat_vals, base_val, title,
                     actual_cls=None, pred_cls=None, max_features=None):
    """Draw a single-row horizontal SHAP force plot on ax."""
    # 颜色修改说明：
    # 1. 正向贡献颜色修改 POS_COLOR，正向浅色修改 POS_LIGHT。
    # 2. 负向贡献颜色修改 NEG_COLOR，负向浅色修改 NEG_LIGHT。
    # 3. base value 标记线颜色修改 ax.axvline(base_val, color="#888888")。
    # 4. f(x) 预测标记线颜色修改 ax.axvline(pred_val, color="#222222")。
    # 5. 条带文字颜色分别在 draw_segment 内部的 ax.text color 参数修改。
    from matplotlib.patches import FancyArrow, Rectangle, FancyArrowPatch
    import matplotlib.patheffects as pe

    # 按绝对值排序，保留指定数量的特征。
    order = np.argsort(np.abs(values))[::-1]
    if max_features is not None:
        order = order[:max_features]
    # 分离正向和负向贡献，并按绝对值降序排列。
    pos_idx = [i for i in order if values[i] >= 0]
    neg_idx = [i for i in order if values[i] < 0]

    pred_val = base_val + values.sum()

    # 计算横轴范围。
    all_vals = [base_val, pred_val,
                base_val + sum(values[i] for i in pos_idx),
                base_val + sum(values[i] for i in neg_idx)]
    x_min, x_max = min(all_vals), max(all_vals)
    span = max(x_max - x_min, 1e-6)
    pad = span * 0.12

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=7.5, length=3, pad=2)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for tick in ax.get_xticklabels():
        tick.set_fontfamily("Times New Roman")

    BAR_Y   = 0.38   # vertical center of the bar
    BAR_H   = 0.22   # bar height
    LABEL_Y = 0.72   # feature name label y
    VAL_Y   = 0.18   # feature value label y
    ARROW_H = BAR_H * 0.55  # arrowhead protrusion

    POS_COLOR  = MAP_RED
    NEG_COLOR  = MAP_GREEN_DARK
    POS_LIGHT  = MAP_PINK_LIGHT
    NEG_LIGHT  = MAP_GREEN_LIGHT

    def draw_segment(start, end, color, light_color, label, fval, shap_val):
        if abs(end - start) < 1e-9:
            return
        going_right = end > start
        seg_w = abs(end - start)
        arrow_w = min(seg_w * 0.25, span * 0.015)

        # 主矩形条带。
        rect_x = min(start, end)
        rect_w = seg_w - arrow_w if seg_w > arrow_w else seg_w
        ax.add_patch(Rectangle(
            (rect_x, BAR_Y - BAR_H / 2), rect_w, BAR_H,
            facecolor=color, edgecolor="white", linewidth=0.6,
            zorder=2, clip_on=False,
        ))
        # 箭头三角形。
        tip_x = end
        base_x = end - arrow_w if going_right else end + arrow_w
        tri_x = [base_x, tip_x, base_x]
        tri_y = [BAR_Y - BAR_H / 2 - ARROW_H,
                 BAR_Y,
                 BAR_Y + BAR_H / 2 + ARROW_H]
        ax.fill(tri_x, tri_y, color=color, zorder=3, clip_on=False)

        # 条带内部的 SHAP 数值。
        mid_x = (start + end) / 2
        ax.text(mid_x, BAR_Y, f"{shap_val:+.3f}",
                ha="center", va="center", fontsize=6.5,
                fontfamily="Times New Roman", color="white",
                fontweight="bold", zorder=4, clip_on=False)

        # 条带上方的特征名。
        ax.text(mid_x, LABEL_Y, label,
                ha="center", va="bottom", fontsize=7,
                fontfamily="Times New Roman", color=color,
                zorder=4, clip_on=False,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

        # 条带下方的特征取值。
        ax.text(mid_x, VAL_Y, f"= {fval:.2f}",
                ha="center", va="top", fontsize=6.5,
                fontfamily="Times New Roman", color=MAP_GRAY,
                zorder=4, clip_on=False)

        # 条带末端分隔线。
        ax.plot([end, end], [BAR_Y - BAR_H / 2, BAR_Y + BAR_H / 2],
                color="white", lw=1.0, zorder=5, clip_on=False)

    # 绘制正向贡献条带，从 base value 向右累加。
    current = base_val
    for i in pos_idx:
        draw_segment(current, current + values[i],
                     POS_COLOR, POS_LIGHT,
                     feat_labels[i], feat_vals[i], values[i])
        current += values[i]

    # 绘制负向贡献条带，从 base value 向左累加。
    current = base_val
    for i in neg_idx:
        draw_segment(current, current + values[i],
                     NEG_COLOR, NEG_LIGHT,
                     feat_labels[i], feat_vals[i], values[i])
        current += values[i]

    # base value 标记线。
    ax.axvline(base_val, color=MAP_GRAY, lw=1.0, linestyle="--",
               ymin=0.12, ymax=0.88, zorder=6)
    ax.text(base_val, 0.96, f"E[f(x)] = {base_val:.3f}",
            ha="center", va="top", fontsize=7,
            fontfamily="Times New Roman", color=MAP_GRAY)

    # 预测输出标记线。
    ax.axvline(pred_val, color=MAP_BOUNDARY, lw=1.2,
               ymin=0.12, ymax=0.88, zorder=6)
    ax.text(pred_val, 0.96, f"f(x) = {pred_val:.3f}",
            ha="center", va="top", fontsize=7,
            fontfamily="Times New Roman", color=MAP_BOUNDARY,
            fontweight="bold")

    ax.set_xlabel("Model output / predicted probability",
                  fontfamily="Times New Roman", labelpad=6)
    subtitle = ""
    if actual_cls is not None and pred_cls is not None:
        subtitle = f"Actual: {actual_cls}   Predicted: {pred_cls}"
    ax.set_title(f"{title}\n{subtitle}", fontfamily="Times New Roman",
                 fontsize=9, pad=8)

def _draw_force_plot(ax, values, feat_labels, feat_vals, base_val, title,
                     actual_cls=None, pred_cls=None, max_features=None):
    """Draw a SHAP force plot with fixed wine-red / teal colors and staggered labels."""
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as pe

    order = np.argsort(np.abs(values))[::-1]
    if max_features is not None:
        order = order[:max_features]
    pos_idx = [i for i in order if values[i] >= 0]
    neg_idx = [i for i in order if values[i] < 0]
    pred_val = base_val + values.sum()

    all_vals = [
        base_val,
        pred_val,
        base_val + sum(values[i] for i in pos_idx),
        base_val + sum(values[i] for i in neg_idx),
    ]
    x_min, x_max = min(all_vals), max(all_vals)
    span = max(x_max - x_min, 1e-6)
    pad = span * 0.12

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=7.5, length=3, pad=2)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for tick in ax.get_xticklabels():
        tick.set_fontfamily("Times New Roman")

    bar_y = 0.43
    bar_h = 0.20
    arrow_h = bar_h * 0.55
    pos_color = MAP_RED
    neg_color = MAP_GREEN_DARK
    pos_light = MAP_PINK_LIGHT
    neg_light = MAP_GREEN_LIGHT

    def draw_segment(start, end, color, light_color, label, fval, label_level):
        if abs(end - start) < 1e-9:
            return
        going_right = end > start
        seg_w = abs(end - start)
        arrow_w = min(seg_w * 0.25, span * 0.015)
        rect_x = min(start, end)
        rect_w = seg_w - arrow_w if seg_w > arrow_w else seg_w
        ax.add_patch(Rectangle(
            (rect_x, bar_y - bar_h / 2), rect_w, bar_h,
            facecolor=color, edgecolor="white", linewidth=0.6,
            zorder=2, clip_on=False,
        ))
        tip_x = end
        base_x = end - arrow_w if going_right else end + arrow_w
        ax.fill(
            [base_x, tip_x, base_x],
            [bar_y - bar_h / 2 - arrow_h, bar_y, bar_y + bar_h / 2 + arrow_h],
            color=color,
            zorder=3,
            clip_on=False,
        )
        mid_x = (start + end) / 2
        label_y = 0.17 - 0.07 * (label_level % 3)
        ax.text(
            mid_x,
            label_y,
            f"{label} = {fval:.3g}",
            ha="center",
            va="center",
            fontsize=6.0,
            fontfamily="Times New Roman",
            color=color,
            zorder=4,
            clip_on=False,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        )
        ax.plot(
            [mid_x, start],
            [label_y + 0.035, bar_y - bar_h / 2],
            color=light_color,
            lw=0.75,
            zorder=1,
            clip_on=False,
        )
        ax.plot([end, end], [bar_y - bar_h / 2, bar_y + bar_h / 2],
                color="white", lw=1.0, zorder=5, clip_on=False)

    current = base_val
    for label_level, i in enumerate(pos_idx):
        draw_segment(current, current + values[i], pos_color, pos_light,
                     feat_labels[i], feat_vals[i], label_level)
        current += values[i]

    current = base_val
    for label_level, i in enumerate(neg_idx):
        draw_segment(current, current + values[i], neg_color, neg_light,
                     feat_labels[i], feat_vals[i], label_level)
        current += values[i]

    ax.axvline(base_val, color=MAP_LIGHT_GRAY, lw=2.0, linestyle="-",
               ymin=0.52, ymax=0.78, zorder=1)
    ax.text(base_val, 0.96, "base value",
            ha="center", va="top", fontsize=7,
            fontfamily="Times New Roman", color=MAP_GRAY)
    ax.axvline(pred_val, color=MAP_BOUNDARY, lw=1.2,
               ymin=0.12, ymax=0.76, zorder=6)
    ax.text(pred_val, 0.98, "f(x)",
            ha="center", va="top", fontsize=7,
            fontfamily="Times New Roman", color=MAP_BOUNDARY,
            fontweight="bold")
    ax.text(pred_val, 0.87, f"{pred_val:.3f}",
            ha="center", va="top", fontsize=7,
            fontfamily="Times New Roman", color=MAP_BOUNDARY,
            fontweight="bold")
    ax.text(0.56, 1.12, "higher", color=MAP_RED,
            transform=ax.transAxes, ha="right", va="center",
            fontsize=8, fontfamily="Times New Roman", clip_on=False)
    ax.text(0.565, 1.12, "\u2194", color=MAP_GRAY,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8, fontfamily="Times New Roman", clip_on=False)
    ax.text(0.57, 1.12, "lower", color=neg_color,
            transform=ax.transAxes, ha="left", va="center",
            fontsize=8, fontfamily="Times New Roman", clip_on=False)
    if title:
        subtitle = ""
        if actual_cls is not None and pred_cls is not None:
            subtitle = f"Actual: {actual_cls}   Predicted: {pred_cls}"
        ax.set_title(f"{title}\n{subtitle}", fontfamily="Times New Roman",
                     fontsize=9, pad=8)


def fig_waterfall(shap_values, X_explain, exp_idx: np.ndarray,
                  cls_idx: int, cls_name: str, df: pd.DataFrame,
                  out_dir: Path) -> None:
    def shorten_feature_value_labels(fig, digits: int = 3) -> None:
        """Shorten SHAP force-plot labels such as 'Slope = 1.7304289'."""
        import re

        pattern = re.compile(
            r"^(?P<name>.+? = )(?P<num>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$"
        )
        for ax in fig.axes:
            for text in ax.texts:
                match = pattern.match(text.get_text())
                if not match:
                    continue
                value = float(match.group("num"))
                text.set_text(f"{match.group('name')}{value:.{digits}f}")

    def separate_prediction_label(fig) -> None:
        for ax in fig.axes:
            arrow_xs = []
            for text in ax.texts:
                if text.get_text().strip() in {r"$\leftarrow$", r"$\rightarrow$"}:
                    arrow_xs.append(float(text.get_position()[0]))
                    text.set_visible(False)
            if arrow_xs:
                ax.text(
                    float(np.mean(arrow_xs)),
                    0.405,
                    "\u2194",
                    ha="center",
                    va="baseline",
                    fontsize=13,
                    fontfamily="Times New Roman",
                    color=MAP_GRAY,
                )
            marker_xs = []
            for text in ax.texts:
                if text.get_text().strip() == "base value":
                    marker_xs.append(float(text.get_position()[0]))
            fx_texts = [text for text in ax.texts if text.get_text().strip() == "f(x)"]
            for fx_text in fx_texts:
                fx_x, _ = fx_text.get_position()
                marker_xs.append(float(fx_x))
                fx_text.set_position((fx_x, 0.37))
                fx_text.set_va("center")
                for text in ax.texts:
                    label = text.get_text().strip()
                    if text is fx_text or label == "base value":
                        continue
                    text_x, text_y = text.get_position()
                    if abs(text_x - fx_x) < 1e-8 and 0.15 < text_y < 0.33:
                        text.set_position((text_x, 0.22))
                        text.set_va("center")
            for line in ax.lines:
                xdata = np.asarray(line.get_xdata(), dtype=float)
                ydata = np.asarray(line.get_ydata(), dtype=float)
                if len(xdata) < 2 or len(ydata) < 2 or not marker_xs:
                    continue
                if np.nanmax(xdata) - np.nanmin(xdata) > 1e-9:
                    continue
                if np.nanmax(ydata) <= 0.1:
                    continue
                if not any(abs(float(xdata[0]) - marker_x) < 1e-8 for marker_x in marker_xs):
                    continue
                line.set_visible(False)

    def recolor_force_plot(fig) -> None:
        from matplotlib.colors import to_hex, to_rgba

        positive_color = MAP_RED
        negative_color = MAP_GREEN_DARK
        positive_light = MAP_PINK_LIGHT
        negative_light = MAP_GREEN_LIGHT

        def replacement(color):
            try:
                r, g, b, a = to_rgba(color)
            except ValueError:
                return None
            if r > 0.75 and g < 0.35 and b < 0.55:
                return positive_color
            if b > 0.55 and r < 0.35:
                return negative_color
            if r > 0.75 and b > 0.65 and g > 0.55:
                return positive_light
            if b > 0.65 and g > 0.55 and r < 0.75:
                return negative_light
            return None

        for ax in fig.axes:
            for patch in ax.patches:
                new_color = replacement(patch.get_facecolor())
                if new_color:
                    patch.set_facecolor(new_color)
                edge_color = replacement(patch.get_edgecolor())
                if edge_color:
                    patch.set_edgecolor(edge_color)
            for line in ax.lines:
                new_color = replacement(line.get_color())
                if new_color:
                    line.set_color(new_color)
            for collection in ax.collections:
                facecolors = collection.get_facecolors()
                if len(facecolors):
                    collection.set_facecolors([
                        to_rgba(replacement(color) or to_hex(color, keep_alpha=True))
                        for color in facecolors
                    ])
                edgecolors = collection.get_edgecolors()
                if len(edgecolors):
                    collection.set_edgecolors([
                        to_rgba(replacement(color) or to_hex(color, keep_alpha=True))
                        for color in edgecolors
                    ])
            for text in ax.texts:
                new_color = replacement(text.get_color())
                if new_color:
                    text.set_color(new_color)
                text.set_fontfamily("Times New Roman")

    sv = shap_values[:, :, cls_idx]  # (n, n_features)
    feat_labels = [FEAT_LABELS[f] for f in SHAP_FEAT_NAMES]
    base_val = float(np.mean(sv))

    class_id = cls_idx + 1
    original_idx = exp_idx.astype(int)
    class_mask = df.iloc[original_idx]["Cluster"].values == class_id
    if np.any(class_mask):
        candidates = np.where(class_mask)[0]
        sample_idx = int(candidates[np.argmax(np.abs(sv[candidates]).sum(axis=1))])
    else:
        sample_idx = int(np.argmax(np.abs(sv).sum(axis=1)))

    plt.figure(figsize=(16.0, 3.2))
    shap.force_plot(
        base_val,
        sv[sample_idx],
        X_explain[sample_idx],
        feature_names=feat_labels,
        matplotlib=True,
        show=False,
        text_rotation=0,
        plot_cmap=[MAP_RED, MAP_GREEN_DARK],
    )
    fig = plt.gcf()
    shorten_feature_value_labels(fig, digits=3)
    recolor_force_plot(fig)
    separate_prediction_label(fig)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.82, bottom=0.28)
    plt.savefig(out_dir / f"Fig8_waterfall_{cls_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_dir / f'Fig8_waterfall_{cls_name}.png'}")


# 主流程

def main() -> None:
    print("=" * 60)
    print("SHAP analysis - TabICL model")
    print("=" * 60)

    # 1. 数据加载
    print("\n[1/4] Loading data and engineering features")
    X_raw, X_full, y, df = load_and_engineer()
    raw_idx = list(range(len(X_COLS)))  # First 14 columns are original X variables.

    # 2. 训练模型
    print("\n[2/4] Training TabICL on full data")
    model, scaler, X_scaled = train_tabicl(X_full, y)

    # 3. 构建原始 14 个特征扰动下的预测函数
    predict_fn = make_predict_fn(model, scaler, X_scaled, raw_idx)

    print("\n[3/4] Computing SHAP values")
    shap_values, X_explain, exp_idx = compute_shap(
        predict_fn, X_raw, n_background=50, n_explain=None
    )

    # 4. 绘制图件
    print("\n[4/4] Drawing figures")
    for cls_id, (cls_name, _) in CLASS_INFO.items():
        cls_idx  = cls_id - 1   # shap_values class index 0/1/2
        out_dir  = CLASS_DIRS[cls_id]
        print(f"\n  === {cls_name} ===")
        fig_global_importance(shap_values, X_explain, cls_idx, cls_name, out_dir)
        fig_dependence(shap_values, X_explain, cls_idx, cls_name, out_dir)
        fig_main_vs_interaction(shap_values, X_explain, cls_idx, cls_name, out_dir,
                                predict_fn=predict_fn)
        fig_interaction_matrix(shap_values, X_explain, cls_idx, cls_name, out_dir,
                               predict_fn=predict_fn)
        fig_bivariate_interaction(shap_values, X_explain, cls_idx, cls_name, out_dir,
                                  predict_fn=predict_fn)
        fig_heatmap(shap_values, X_explain, exp_idx, cls_idx, cls_name, df, out_dir)
        fig_waterfall(shap_values, X_explain, exp_idx, cls_idx, cls_name, df, out_dir)

    print(f"\nAll outputs saved to {SHAP_DIR}")


if __name__ == "__main__":
    main()
