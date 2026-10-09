"""
村落形态演化分析 - 主入口
用法：
    python scripts/main.py --xrfm          # 训练 xRFM（固定划分）
    python scripts/main.py --xrfm --cv     # 训练 xRFM（GroupKFold CV，推荐）
    python scripts/main.py --all --cv      # 所有模型 CV 评估并汇总
    python scripts/main.py --lr            # 逻辑回归
    python scripts/main.py --dt            # 决策树
    python scripts/main.py --rf            # 随机森林
    python scripts/main.py --xgboost       # XGBoost
    python scripts/main.py --lightgbm      # LightGBM
    python scripts/main.py --tabm          # TabM
    python scripts/main.py --tabpfn        # TabPFN v2
    python scripts/main.py --tune-xrfm     # xRFM 超参数网格搜索
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR      = PROJECT_ROOT / "src"
XRFM_DIR     = PROJECT_ROOT / "xRFM"

for _p in (str(SRC_DIR), str(XRFM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────── 常量 ───────────────────────────
X_COLS = [
    "ELEVATION", "SLOPE", "ASPECT", "MAP", "MAT",
    "NDVI", "DWS", "RD", "POP", "PGDP", "NLI",
    "CRPU", "ICHD", "DAR",
]
Y_COLS = ["MBC", "LCC", "GCC", "INT", "CSR", "IR", "AEL"]
YEARS = [2014, 2016, 2018, 2020]
YEAR_TO_IDX = {y: i for i, y in enumerate(YEARS)}

ALL_MODELS = ["lr", "dt", "rf", "xgboost", "lightgbm", "tabm", "tabpfn", "tabicl", "xrfm"]


# ─────────────────────────── 数据加载 ───────────────────────────

def load_data(x_path: Path, cluster_path: Path) -> pd.DataFrame:
    """合并自变量、形态指标与聚类标签，返回完整面板。"""
    x_df = pd.read_excel(x_path)
    c_df = pd.read_excel(cluster_path, usecols=["ID", "YEAR", "Cluster"] + Y_COLS)

    x_df["ID"] = x_df["ID"].astype(str).str.strip()
    c_df["ID"] = c_df["ID"].astype(str).str.strip()
    x_df["YEAR"] = x_df["YEAR"].astype(int)
    c_df["YEAR"] = c_df["YEAR"].astype(int)

    df = x_df.merge(c_df, on=["ID", "YEAR"], how="inner")
    if df.empty:
        raise ValueError("合并后数据为空，请检查 ID/YEAR 是否匹配。")

    df["VILLAGE_ID"] = df["ID"].str.replace(r"(2014|2016|2018|2020)$", "", regex=True)

    missing = df[X_COLS].isna().sum().sum()
    if missing:
        print(f"[警告] 自变量存在 {missing} 个缺失值，用列中位数填充。")
        df[X_COLS] = df[X_COLS].fillna(df[X_COLS].median(numeric_only=True))

    print(f"[数据] 合并后共 {len(df)} 条记录，{df['VILLAGE_ID'].nunique()} 个村落")
    print(f"  聚类分布：\n{df['Cluster'].value_counts().sort_index().to_string()}")
    return df


# ─────────────────────────── 数据划分（按村落）───────────────────────────

def split_by_village(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    village_df = (
        df.groupby("VILLAGE_ID")["Cluster"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
        .rename(columns={"Cluster": "majority_cluster"})
    )

    villages = village_df["VILLAGE_ID"].values
    labels   = village_df["majority_cluster"].values

    test_villages, val_villages, train_villages = [], [], []
    for cls in np.unique(labels):
        cls_v = villages[labels == cls].copy()
        rng.shuffle(cls_v)
        n = len(cls_v)
        n_test = max(1, round(n * test_size))
        n_val  = max(1, round(n * val_size))
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train = 1
            n_val = max(0, n - n_test - n_train)
        test_villages.extend(cls_v[:n_test].tolist())
        val_villages.extend(cls_v[n_test:n_test + n_val].tolist())
        train_villages.extend(cls_v[n_test + n_val:].tolist())

    train_df = df[df["VILLAGE_ID"].isin(train_villages)].copy()
    val_df   = df[df["VILLAGE_ID"].isin(val_villages)].copy()
    test_df  = df[df["VILLAGE_ID"].isin(test_villages)].copy()

    print(f"[划分] 村落数 train={len(train_villages)}, val={len(val_villages)}, test={len(test_villages)}")
    print(f"       记录数 train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    for name, sub in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = sub["Cluster"].value_counts().sort_index().to_dict()
        print(f"  {name} 类别分布: {dist}")

    return train_df, val_df, test_df


# ─────────────────────────── 特征工程 ───────────────────────────

def _compute_diff_feats(df: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    diff_feats = np.zeros_like(raw)
    df_reset = df.reset_index(drop=True)
    for _, grp in df_reset.groupby("VILLAGE_ID"):
        idx = grp.index.tolist()
        vals = raw[idx]
        diff_feats[idx] = np.diff(vals, axis=0, prepend=vals[:1])
    return diff_feats


def _compute_lag_y(df: pd.DataFrame, y_vals: np.ndarray) -> np.ndarray:
    lag = np.zeros_like(y_vals)
    df_reset = df.reset_index(drop=True)
    for _, grp in df_reset.groupby("VILLAGE_ID"):
        idx = grp.index.tolist()
        vals = y_vals[idx]
        lag[idx[0]] = vals[0]
        for i in range(1, len(idx)):
            lag[idx[i]] = vals[i - 1]
    return lag


def build_features_for_split(
    df: pd.DataFrame,
    train_village_stats: dict | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    df = df.copy().sort_values(["VILLAGE_ID", "YEAR"]).reset_index(drop=True)

    raw      = df[X_COLS].values.astype(np.float32)
    y_vals   = df[Y_COLS].values.astype(np.float32)
    year_idx = df["YEAR"].map(YEAR_TO_IDX).values.astype(np.float32)
    year_norm = year_idx / (len(YEARS) - 1)
    year_sin  = np.sin(2 * np.pi * year_norm)
    year_cos  = np.cos(2 * np.pi * year_norm)
    time_feats   = np.column_stack([year_norm, year_sin, year_cos])
    interaction  = raw * year_norm[:, None]
    diff_feats   = _compute_diff_feats(df, raw)
    lag_y        = _compute_lag_y(df, y_vals)

    mean_feats = np.zeros_like(raw)
    std_feats  = np.zeros_like(raw)

    if train_village_stats is None:
        village_stats: dict = {}
        global_mean = raw.mean(axis=0)
        global_std  = raw.std(axis=0)
        for vid, grp in df.groupby("VILLAGE_ID"):
            idx = grp.index.tolist()
            vals = raw[idx]
            village_stats[vid] = {"mean": vals.mean(axis=0), "std": vals.std(axis=0)}
            mean_feats[idx] = village_stats[vid]["mean"]
            std_feats[idx]  = village_stats[vid]["std"]
        returned_stats = {"village": village_stats, "global_mean": global_mean, "global_std": global_std}
    else:
        village_stats = train_village_stats["village"]
        global_mean   = train_village_stats["global_mean"]
        global_std    = train_village_stats["global_std"]
        returned_stats = train_village_stats
        for vid, grp in df.groupby("VILLAGE_ID"):
            idx = grp.index.tolist()
            if vid in village_stats:
                mean_feats[idx] = village_stats[vid]["mean"]
                std_feats[idx]  = village_stats[vid]["std"]
            else:
                mean_feats[idx] = global_mean
                std_feats[idx]  = global_std

    X = np.concatenate(
        [raw, time_feats, interaction, diff_feats, lag_y, mean_feats, std_feats],
        axis=1,
    ).astype(np.float32)

    y = df["Cluster"].values.astype(np.int64)
    return X, y, returned_stats


def prepare_data(args: argparse.Namespace):
    """加载、划分、特征工程、标准化，返回固定 train/val/test 数据。"""
    df = load_data(
        x_path=PROJECT_ROOT / args.x_file,
        cluster_path=PROJECT_ROOT / args.cluster_file,
    )
    train_df, val_df, test_df = split_by_village(
        df, val_size=args.val_size, test_size=args.test_size, random_state=args.seed,
    )
    X_train, y_train, train_stats = build_features_for_split(train_df, None)
    X_val,   y_val,   _           = build_features_for_split(val_df,   train_stats)
    X_test,  y_test,  _           = build_features_for_split(test_df,  train_stats)

    print(f"[特征工程] 特征维度：{X_train.shape[1]}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test, test_df


def prepare_data_cv(args: argparse.Namespace):
    """加载全量数据并构造特征，返回用于 GroupKFold CV 的矩阵。"""
    df = load_data(
        x_path=PROJECT_ROOT / args.x_file,
        cluster_path=PROJECT_ROOT / args.cluster_file,
    )
    df = df.copy().sort_values(["VILLAGE_ID", "YEAR"]).reset_index(drop=True)

    raw    = df[X_COLS].values.astype(np.float32)
    y_vals = df[Y_COLS].values.astype(np.float32)
    year_idx  = df["YEAR"].map(YEAR_TO_IDX).values.astype(np.float32)
    year_norm = year_idx / (len(YEARS) - 1)
    year_sin  = np.sin(2 * np.pi * year_norm)
    year_cos  = np.cos(2 * np.pi * year_norm)
    time_feats  = np.column_stack([year_norm, year_sin, year_cos])
    interaction = raw * year_norm[:, None]

    diff_x = np.zeros_like(raw)
    lag_y  = np.zeros_like(y_vals)
    mean_feats = np.zeros_like(raw)
    std_feats  = np.zeros_like(raw)

    for vid, grp in df.groupby("VILLAGE_ID"):
        idx = grp.index.tolist()
        vals = raw[idx]
        diff_x[idx] = np.diff(vals, axis=0, prepend=vals[:1])
        lag_y[idx[0]] = y_vals[idx[0]]
        for i in range(1, len(idx)):
            lag_y[idx[i]] = y_vals[idx[i - 1]]
        mean_feats[idx] = vals.mean(axis=0)
        std_feats[idx]  = vals.std(axis=0)

    # 三种特征子集（与老师给的代码对应）
    feature_sets = {
        "raw_14": raw,
        "dynamic_73": np.concatenate(
            [raw, time_feats, interaction, diff_x, mean_feats, std_feats], axis=1
        ),
        "full_80": np.concatenate(
            [raw, time_feats, interaction, diff_x, lag_y, mean_feats, std_feats], axis=1
        ),
    }

    X = feature_sets["full_80"]
    y      = df["Cluster"].values.astype(np.int64)
    groups = df["VILLAGE_ID"].values

    # 注意：CV 模式下 StandardScaler 在每折内部单独拟合，这里不做全局标准化
    print(f"[特征工程] 特征维度：{X.shape[1]}，样本数：{X.shape[0]}")
    return X, y, groups, df, feature_sets



# ─────────────────────────── 评估（xRFM 专用）───────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, split_name: str,
             y_proba: np.ndarray | None = None) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report, confusion_matrix, roc_auc_score,
    )
    acc                = accuracy_score(y_true, y_pred)
    precision_macro    = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_macro       = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    recall_weighted    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro           = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    auc_ovr, auc_ovo = None, None
    if y_proba is not None:
        try:
            auc_ovr = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            auc_ovo = roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro")
        except Exception:
            pass

    print(f"\n[{split_name}]")
    print(f"  Accuracy          = {acc:.4f}")
    print(f"  Precision (macro) = {precision_macro:.4f}  Precision (weighted) = {precision_weighted:.4f}")
    print(f"  Recall    (macro) = {recall_macro:.4f}  Recall    (weighted) = {recall_weighted:.4f}")
    print(f"  F1        (macro) = {f1_macro:.4f}  F1        (weighted) = {f1_weighted:.4f}")
    if auc_ovr is not None:
        print(f"  AUC (OvR macro)   = {auc_ovr:.4f}  AUC (OvO macro)   = {auc_ovo:.4f}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("混淆矩阵：")
    print(confusion_matrix(y_true, y_pred))

    return {
        "split":               split_name,
        "accuracy":            acc,
        "precision_macro":     precision_macro,
        "precision_weighted":  precision_weighted,
        "recall_macro":        recall_macro,
        "recall_weighted":     recall_weighted,
        "f1_macro":            f1_macro,
        "f1_weighted":         f1_weighted,
        "auc_ovr":             auc_ovr,
        "auc_ovo":             auc_ovo,
    }


# ─────────────────────────── xRFM ───────────────────────────

def run_xrfm(
    X_train, y_train, X_val, y_val, X_test, y_test,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    import torch
    from xrfm import xRFM
    from models.base import save_results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[xRFM] 训练 xRFM … (设备: {device})")

    Xt = torch.from_numpy(X_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    Xs = torch.from_numpy(X_test).to(device)
    yt = torch.from_numpy(y_train).long()
    yv = torch.from_numpy(y_val).long()

    rfm_params = {
        "model": {"kernel": args.kernel, "bandwidth": args.bandwidth,
                  "exponent": 1.0, "diag": False, "bandwidth_mode": "constant"},
        "fit":   {"reg": args.reg, "iters": args.iters, "early_stop_rfm": True,
                  "early_stop_multiplier": args.early_stop_multiplier},
    }
    default_rfm_params = {
        "model": {"kernel": "l2_high_dim", "exponent": 1.0, "bandwidth": 10.0,
                  "diag": False, "bandwidth_mode": "constant"},
        "fit":   {"get_agop_best_model": True, "return_best_params": False,
                  "reg": 1e-3, "iters": 0, "early_stop_rfm": False},
    }

    model = xRFM(
        rfm_params=rfm_params, device=device, tuning_metric="accuracy",
        max_leaf_size=args.max_leaf_size, n_trees=args.n_trees,
        n_tree_iters=args.n_tree_iters, split_method=args.split_method,
        default_rfm_params=default_rfm_params, random_state=args.seed, verbose=True,
    )
    model.fit(Xt, yt, Xv, yv)

    y_pred_train = model.predict(Xt)
    y_pred_val   = model.predict(Xv)
    y_pred_test  = model.predict(Xs)

    train_metrics = evaluate(y_train, y_pred_train, "训练集")
    val_metrics   = evaluate(y_val,   y_pred_val,   "验证集")
    test_metrics  = evaluate(y_test,  y_pred_test,  "测试集")

    out_dir = PROJECT_ROOT / args.output_dir / "xrfm"
    save_results(out_dir, "xrfm", train_metrics, val_metrics, test_metrics,
                 test_df, y_test, y_pred_test)

    import torch as _torch
    state = model.get_state_dict()
    _torch.save(state, out_dir / "xrfm_state.pt")
    print(f"[输出] 模型状态已保存至 {out_dir / 'xrfm_state.pt'}")

    return test_metrics


# ─────────────────────────── 汇总 ───────────────────────────

def save_summary(all_results: dict[str, dict], out_dir: Path) -> None:
    rows = []
    for model_name, metrics in all_results.items():
        rows.append({"model": model_name, **{k: v for k, v in metrics.items() if k != "split"}})
    summary = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "all_models_summary.xlsx"
    summary.to_excel(path, index=False)

    print("\n" + "=" * 70)
    print("模型对比汇总（按测试集 F1-macro 降序）")
    print("=" * 70)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[输出] 汇总结果已保存至 {path}")


# ─────────────────────────── GroupKFold CV 训练 ───────────────────────────

def run_cv(model_name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
           args: argparse.Namespace, n_splits: int = 5,
           df: pd.DataFrame | None = None) -> dict:
    """
    用 GroupKFold CV 训练并评估单个模型。
    每折内部：用训练集拟合 StandardScaler，val 集（20%训练集）用于早停/调参。
    返回各指标在所有折上的均值和标准差。
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    all_te_idx, all_y_true, all_y_pred = [], [], []

    print(f"\n[{model_name.upper()}] GroupKFold CV（{n_splits}折）…")

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
        X_te,     y_te = X[te_idx], y[te_idx]

        # 每折内部标准化
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_raw).astype(np.float32)
        X_te_sc = scaler.transform(X_te).astype(np.float32)

        # 从训练集切出 20% 作为 val（用于早停）
        n_val   = max(10, int(len(X_tr_sc) * 0.2))
        X_val_sc, y_val = X_tr_sc[-n_val:], y_tr[-n_val:]
        X_fit,    y_fit = X_tr_sc[:-n_val], y_tr[:-n_val]

        import io, contextlib
        if model_name == "xrfm":
            import torch
            from xrfm import xRFM
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            rfm_p = {
                "model": {"kernel": args.kernel, "bandwidth": args.bandwidth,
                          "exponent": 1.0, "diag": False, "bandwidth_mode": "constant"},
                "fit":   {"reg": args.reg, "iters": args.iters, "early_stop_rfm": True,
                          "early_stop_multiplier": args.early_stop_multiplier},
            }
            m = xRFM(rfm_params=rfm_p, device=device, tuning_metric="accuracy",
                     max_leaf_size=args.max_leaf_size, n_trees=args.n_trees,
                     n_tree_iters=args.n_tree_iters, split_method=args.split_method,
                     random_state=args.seed, verbose=False)
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(torch.from_numpy(X_fit).to(device),    torch.from_numpy(y_fit).long(),
                      torch.from_numpy(X_val_sc).to(device), torch.from_numpy(y_val).long())
            y_pred_tr = m.predict(torch.from_numpy(X_fit).to(device))
            y_pred    = m.predict(torch.from_numpy(X_te_sc).to(device))
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                m, y_pred = _fit_sklearn_model(
                    model_name, X_fit, y_fit, X_val_sc, y_val, X_te_sc,
                    seed=args.seed + fold,
                )
            if m is None:
                # 模型跳过（如 TabPFN license 未接受），打印提示后跳过本折
                print(f"  Fold {fold}: 跳过（模型不可用）")
                continue
            # 训练集预测（tabm 返回的 m 是 torch 模块，需特殊处理）
            if model_name == "tabm":
                from models.tabm.model import _predict, _predict_proba
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                y_pred_tr  = _predict(m, X_fit, device) + 1
                te_proba   = _predict_proba(m, X_te_sc, device)
            elif model_name in ("xgboost", "lightgbm"):
                y_pred_tr = m.predict(X_fit) + 1
                te_proba  = m.predict_proba(X_te_sc)
            else:
                y_pred_tr = m.predict(X_fit)
                te_proba  = m.predict_proba(X_te_sc) if hasattr(m, "predict_proba") else None

        # xRFM 概率
        if model_name == "xrfm":
            import torch
            _proba = m.predict_proba(torch.from_numpy(X_te_sc).to(device))
            if hasattr(_proba, "cpu"):
                _proba = _proba.cpu().numpy()
            # xRFM predict_proba 输出列数 = max_label+1，第0列对应标签0（不存在）
            # 标签为 1/2/3，取第1~3列
            n_classes = len(np.unique(y_te))
            te_proba = _proba[:, 1:1 + n_classes]
            # 重新归一化，防止数值误差
            te_proba = te_proba / te_proba.sum(axis=1, keepdims=True)

        # 训练集指标
        tr_acc  = accuracy_score(y_fit, y_pred_tr)
        tr_f1   = f1_score(y_fit, y_pred_tr, average="macro", zero_division=0)
        # 收集预测结果（用于事后保存 test_predictions）
        all_te_idx.extend(te_idx.tolist())
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred))
        # 测试折指标
        from sklearn.metrics import roc_auc_score
        te_acc  = accuracy_score(y_te, y_pred)
        te_prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
        te_rec  = recall_score(y_te, y_pred, average="macro", zero_division=0)
        te_f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)
        te_auc_ovr, te_auc_ovo = None, None
        if te_proba is not None:
            try:
                te_auc_ovr = roc_auc_score(y_te, te_proba, multi_class="ovr", average="macro")
                te_auc_ovo = roc_auc_score(y_te, te_proba, multi_class="ovo", average="macro")
            except Exception:
                pass

        fold_metrics.append({
            "fold":              fold,
            "train_accuracy":    tr_acc,
            "train_f1_macro":    tr_f1,
            "test_accuracy":     te_acc,
            "test_precision_macro": te_prec,
            "test_recall_macro": te_rec,
            "test_f1_macro":     te_f1,
            "test_auc_ovr":      te_auc_ovr,
            "test_auc_ovo":      te_auc_ovo,
        })
        auc_str = f"  AUC_OvR={te_auc_ovr:.4f}  AUC_OvO={te_auc_ovo:.4f}" if te_auc_ovr is not None else ""
        print(f"  Fold {fold}:  Train Acc={tr_acc:.4f}  Train F1={tr_f1:.4f}"
              f"  |  Test Acc={te_acc:.4f}  Test Prec={te_prec:.4f}"
              f"  Test Rec={te_rec:.4f}  Test F1={te_f1:.4f}{auc_str}")

    # 汇总
    if not fold_metrics:
        print(f"  [{model_name.upper()}] 所有折均跳过，模型不可用。")
        return {}

    fm = pd.DataFrame(fold_metrics)
    tr_means = fm[["train_accuracy", "train_f1_macro"]].mean()
    te_cols  = ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]
    te_means = fm[te_cols].mean()
    te_stds  = fm[te_cols].std()

    has_auc = fm["test_auc_ovr"].notna().any()
    auc_means = fm[["test_auc_ovr", "test_auc_ovo"]].mean() if has_auc else None
    auc_stds  = fm[["test_auc_ovr", "test_auc_ovo"]].std()  if has_auc else None

    print(f"\n  训练集均值: Accuracy={tr_means['train_accuracy']:.4f}  F1-macro={tr_means['train_f1_macro']:.4f}")
    print(f"  测试折均值: Accuracy={te_means['test_accuracy']:.4f}  "
          f"Precision={te_means['test_precision_macro']:.4f}  "
          f"Recall={te_means['test_recall_macro']:.4f}  "
          f"F1-macro={te_means['test_f1_macro']:.4f}")
    if has_auc:
        print(f"             AUC(OvR)={auc_means['test_auc_ovr']:.4f}  AUC(OvO)={auc_means['test_auc_ovo']:.4f}")
    print(f"  测试折标准差: Accuracy={te_stds['test_accuracy']:.4f}  "
          f"F1-macro={te_stds['test_f1_macro']:.4f}")

    return {
        "split":                 "cv_mean",
        "train_accuracy":        float(tr_means["train_accuracy"]),
        "train_f1_macro":        float(tr_means["train_f1_macro"]),
        "accuracy":              float(te_means["test_accuracy"]),
        "precision_macro":       float(te_means["test_precision_macro"]),
        "recall_macro":          float(te_means["test_recall_macro"]),
        "f1_macro":              float(te_means["test_f1_macro"]),
        "auc_ovr":               float(auc_means["test_auc_ovr"]) if has_auc else None,
        "auc_ovo":               float(auc_means["test_auc_ovo"]) if has_auc else None,
        "accuracy_std":          float(te_stds["test_accuracy"]),
        "precision_std":         float(te_stds["test_precision_macro"]),
        "recall_std":            float(te_stds["test_recall_macro"]),
        "f1_macro_std":          float(te_stds["test_f1_macro"]),
        "auc_ovr_std":           float(auc_stds["test_auc_ovr"]) if has_auc else None,
        "auc_ovo_std":           float(auc_stds["test_auc_ovo"]) if has_auc else None,
        "fold_details":          fold_metrics,
        "all_te_idx":            all_te_idx,
        "all_y_true":            all_y_true,
        "all_y_pred":            all_y_pred,
    }


def _save_cv_predictions(model_name: str, all_te_idx: list, all_y_true: list,
                         all_y_pred: list, df: pd.DataFrame | None,
                         out_dir: Path, feature_set: str = "full_80") -> None:
    """将 CV 各折的测试预测拼合保存为 test_predictions.xlsx。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({"idx": all_te_idx, "y_true": all_y_true, "y_pred": all_y_pred})
    pred_df = pred_df.sort_values("idx").reset_index(drop=True)

    if df is not None:
        pred_df["VILLAGE_ID"] = df.iloc[pred_df["idx"].values]["VILLAGE_ID"].values
        pred_df["YEAR"]       = df.iloc[pred_df["idx"].values]["YEAR"].values
        pred_df = pred_df[["VILLAGE_ID", "YEAR", "y_true", "y_pred"]]
    else:
        pred_df = pred_df[["y_true", "y_pred"]]

    path = out_dir / f"{model_name}_{feature_set}_test_predictions.xlsx"
    pred_df.to_excel(path, index=False)
    print(f"[输出] CV 预测已保存至 {path}")


def _run_with_timeout(fn, timeout: float, desc: str):
    """跨平台超时包装。

    Windows 没有 signal.SIGALRM，原代码用 signal.alarm 会直接抛异常、
    被 except 吞掉导致模型被静默跳过。这里改用线程池实现超时控制。
    超时后抛出 TimeoutError（仅放弃等待，后台线程可能被丢弃，但模型对象不再使用）。
    """
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"{desc} 超过 {timeout}s 仍未完成")


def _fit_sklearn_model(model_name: str, X_fit, y_fit, X_val, y_val, X_te, seed: int):
    """为各 sklearn 系模型统一封装 fit + predict。"""
    if model_name == "lr":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)
        m.fit(X_fit, y_fit)

    elif model_name == "dt":
        from sklearn.tree import DecisionTreeClassifier
        best_depth, best_acc = None, -1
        for depth in [3, 5, 7, 10, None]:
            tmp = DecisionTreeClassifier(max_depth=depth, random_state=seed)
            tmp.fit(X_fit, y_fit)
            acc = (tmp.predict(X_val) == y_val).mean()
            if acc > best_acc:
                best_acc, best_depth = acc, depth
        m = DecisionTreeClassifier(max_depth=best_depth, random_state=seed)
        m.fit(X_fit, y_fit)

    elif model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        best_n, best_acc = 100, -1
        for n in [50, 100, 200, 300]:
            tmp = RandomForestClassifier(
                n_estimators=n, max_depth=8, min_samples_leaf=4,
                max_features="sqrt", random_state=seed, n_jobs=-1,
            )
            tmp.fit(X_fit, y_fit)
            acc = (tmp.predict(X_val) == y_val).mean()
            if acc > best_acc:
                best_acc, best_n = acc, n
        m = RandomForestClassifier(
            n_estimators=best_n, max_depth=8, min_samples_leaf=4,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        )
        m.fit(X_fit, y_fit)

    elif model_name == "xgboost":
        import xgboost as xgb
        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.01,
            subsample=0.5, colsample_bytree=0.4,
            min_child_weight=15, gamma=3.0, reg_alpha=2.0, reg_lambda=10.0,
            eval_metric="mlogloss", early_stopping_rounds=20,
            random_state=seed, verbosity=0,
        )
        m.fit(X_fit, y_fit - 1, eval_set=[(X_val, y_val - 1)], verbose=False)
        return m, m.predict(X_te) + 1

    elif model_name == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(
            n_estimators=200, max_depth=2, num_leaves=6,
            learning_rate=0.01, subsample=0.5, colsample_bytree=0.4,
            min_child_samples=30, reg_alpha=2.0, reg_lambda=10.0,
            random_state=seed, verbose=-1,
        )
        m.fit(X_fit, y_fit - 1,
              eval_set=[(X_val, y_val - 1)],
              callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
        return m, m.predict(X_te) + 1

    elif model_name == "tabm":
        import sys as _sys
        _sys.path.insert(0, str(SRC_DIR))
        from models.tabm.model import _build_model, _train, _predict
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        n_classes = len(np.unique(y_fit))
        mod = _build_model(X_fit.shape[1], n_classes)
        mod = _train(mod, X_fit, y_fit - 1, X_val, y_val - 1, device,
                     epochs=200, lr=2e-4, patience=20, weight_decay=5e-3)
        return mod, _predict(mod, X_te, device) + 1

    elif model_name == "tabpfn":
        try:
            import os
            from tabpfn import TabPFNClassifier
            # 支持通过环境变量提供 license token（非交互环境，如服务器/Windows）
            token = os.environ.get("TABPFN_TOKEN")
            if not token:
                # 兼容 Windows cmd 无法可靠传递环境变量的场景：从本地文件读取
                _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                _tok_file = os.path.join(_root, "tabpfn_token.txt")
                if os.path.exists(_tok_file):
                    with open(_tok_file, "r", encoding="utf-8") as _f:
                        token = _f.read().strip()
            if token:
                os.environ["TABPFN_TOKEN"] = token
            else:
                # 无 token 时 TabPFN 会在 fit 阶段尝试联网下载权重并卡住，
                # 这里直接跳过，避免整个 CV 流程被阻塞。
                print("[TabPFN] 跳过：未设置 TABPFN_TOKEN（需先接受 license 并配置 API Key）。")
                print("  1. 浏览器打开 https://ux.priorlabs.ai 登录并勾选 License 同意")
                print("  2. 复制 API Key，运行前设置：set TABPFN_TOKEN=<your-api-key>")
                return None, np.zeros(len(X_te), dtype=np.int64)
            # 只用一半训练数据，限制 TabPFN 的 in-context 信息量
            rng = np.random.default_rng(seed)
            half = max(20, len(X_fit) // 2)
            idx = rng.choice(len(X_fit), half, replace=False)
            X_sub = np.concatenate([X_fit[idx], X_val], axis=0)
            y_sub = np.concatenate([y_fit[idx], y_val], axis=0)
            m = TabPFNClassifier(n_estimators=4, random_state=seed, ignore_pretraining_limits=True)
            try:
                _run_with_timeout(lambda: m.fit(X_sub, y_sub), 900, "TabPFN fit")
            except Exception as _e:
                import traceback as _tb
                _err = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "outputs", "tabpfn_err.txt"), "a", encoding="utf-8")
                _err.write(f"[fold seed={seed}] EXC {type(_e).__name__}: {_e}\n")
                _tb.print_exc(file=_err)
                _err.write("=" * 60 + "\n")
                _err.flush(); _err.close()
                raise
        except TimeoutError as e:
            print(f"[TabPFN] 跳过：{e}（可能正在下载模型权重）")
            return None, np.zeros(len(X_te), dtype=np.int64)
        except Exception as e:
            ename = type(e).__name__
            if "License" in ename or "license" in str(e).lower():
                print("[TabPFN] 跳过：需要先接受 license 并配置 TABPFN_TOKEN。")
                print("  1. 浏览器打开 https://ux.priorlabs.ai 登录并勾选 License 同意")
                print("  2. 复制 API Key，设置环境变量后重试：")
                print("       set TABPFN_TOKEN=<your-api-key>   (Windows cmd)")
                print("     或 Python 中：import os; os.environ['TABPFN_TOKEN']='<key>'")
            else:
                print(f"[TabPFN] 跳过：{e}")
            return None, np.zeros(len(X_te), dtype=np.int64)
        # fit 成功：返回模型与测试集预测（缺失此返回会导致上层误判为“模型不可用”）
        return m, m.predict(X_te)

    elif model_name == "tabicl":
        try:
            from tabicl import TabICLClassifier
            X_full = np.concatenate([X_fit, X_val], axis=0)
            y_full = np.concatenate([y_fit, y_val], axis=0)
            m = TabICLClassifier(n_estimators=8, random_state=seed)
            _run_with_timeout(lambda: m.fit(X_full, y_full), 900, "TabICLv2 fit")
        except TimeoutError as e:
            print(f"[TabICLv2] 跳过：{e}（可能正在下载模型权重）")
            return None, np.zeros(len(X_te), dtype=np.int64)
        except Exception as e:
            print(f"[TabICLv2] 跳过：{e}")
            return None, np.zeros(len(X_te), dtype=np.int64)

    else:
        raise ValueError(f"未知模型: {model_name}")

    return m, m.predict(X_te)


# ─────────────────────────── xRFM 超参数搜索 ───────────────────────────

def tune_xrfm(args: argparse.Namespace) -> None:
    import torch
    from itertools import product
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score, accuracy_score
    from xrfm import xRFM

    print("\n[xRFM 超参数搜索] 加载数据（完整 80 维特征）…")
    _, y, groups, _, feature_sets = prepare_data_cv(args)
    X_full_raw = feature_sets["full_80"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gkf    = GroupKFold(n_splits=5)

    param_grid = {
        "bandwidth": [0.5, 1.0, 2.0, 5.0, 10.0],
        "reg":       [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "iters":     [3, 5, 7],
        "kernel":    ["sum_power_laplace", "l2_high_dim"],
    }
    combos = list(product(
        param_grid["bandwidth"],
        param_grid["reg"],
        param_grid["iters"],
        param_grid["kernel"],
    ))
    print(f"[xRFM 超参数搜索] 共 {len(combos)} 个组合，设备: {device}")
    print("%-6s %-6s %-6s %-22s  f1_macro  accuracy" % ("bw", "reg", "iters", "kernel"))
    print("-" * 65)

    results = []
    best_f1, best_cfg = -1.0, None

    for bw, reg, iters, kernel in combos:
        fold_f1, fold_acc = [], []
        for tr_idx, te_idx in gkf.split(X_full_raw, y, groups):
            X_tr_raw, y_tr = X_full_raw[tr_idx], y[tr_idx]
            X_te_raw, y_te = X_full_raw[te_idx],  y[te_idx]

            # 每折内部标准化（与 run_cv 保持一致）
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw).astype(np.float32)
            X_te = scaler.transform(X_te_raw).astype(np.float32)

            n_val  = max(10, int(len(X_tr) * 0.2))
            X_v,  y_v  = X_tr[-n_val:], y_tr[-n_val:]
            X_t2, y_t2 = X_tr[:-n_val], y_tr[:-n_val]

            rfm_p = {
                "model": {"kernel": kernel, "bandwidth": bw, "exponent": 1.0,
                          "diag": False, "bandwidth_mode": "constant"},
                "fit":   {"reg": reg, "iters": iters, "early_stop_rfm": True},
            }
            m = xRFM(rfm_params=rfm_p, device=device, tuning_metric="accuracy",
                     max_leaf_size=60_000, verbose=False, random_state=args.seed)
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(torch.from_numpy(X_t2).to(device), torch.from_numpy(y_t2).long(),
                      torch.from_numpy(X_v).to(device),  torch.from_numpy(y_v).long())
            yp = m.predict(torch.from_numpy(X_te).to(device))
            fold_f1.append(f1_score(y_te, yp, average="macro"))
            fold_acc.append(accuracy_score(y_te, yp))

        mf1 = float(np.mean(fold_f1))
        mac = float(np.mean(fold_acc))
        marker = " <-- best" if mf1 > best_f1 else ""
        print("%-6s %-6s %-6s %-22s  %.4f    %.4f%s" % (bw, reg, iters, kernel, mf1, mac, marker), flush=True)
        results.append({"bandwidth": bw, "reg": reg, "iters": iters, "kernel": kernel,
                        "f1_macro": mf1, "accuracy": mac})
        if mf1 > best_f1:
            best_f1  = mf1
            best_cfg = {"bandwidth": bw, "reg": reg, "iters": iters, "kernel": kernel}

    # 保存搜索结果
    out_dir = PROJECT_ROOT / args.output_dir / "xrfm"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    results_path = out_dir / "xrfm_tune_results.xlsx"
    results_df.to_excel(results_path, index=False)

    print("\n" + "=" * 65)
    print("最优参数组合：")
    for k, v in best_cfg.items():
        print(f"  {k} = {v}")
    print(f"最优 GroupKFold F1-macro = {best_f1:.4f}")
    print(f"\n[输出] 搜索结果已保存至 {results_path}")
    print("提示：将上述参数传入 --xrfm 使用，例如：")
    print(f"  python scripts/main.py --xrfm "
          f"--kernel {best_cfg['kernel']} "
          f"--bandwidth {best_cfg['bandwidth']} "
          f"--reg {best_cfg['reg']} "
          f"--iters {best_cfg['iters']}")


# ─────────────────────────── main ───────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="村落形态演化分析主入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型选择
    parser.add_argument("--all",        action="store_true", help="训练所有模型并汇总")
    parser.add_argument("--lr",         action="store_true", help="逻辑回归")
    parser.add_argument("--dt",         action="store_true", help="决策树")
    parser.add_argument("--rf",         action="store_true", help="随机森林")
    parser.add_argument("--xgboost",    action="store_true", help="XGBoost")
    parser.add_argument("--lightgbm",   action="store_true", help="LightGBM")
    parser.add_argument("--tabm",       action="store_true", help="TabM")
    parser.add_argument("--tabpfn",     action="store_true", help="TabPFN v2")
    parser.add_argument("--tabicl",     action="store_true", help="TabICL v2")
    parser.add_argument("--xrfm",       action="store_true", help="xRFM")
    parser.add_argument("--tune-xrfm",  action="store_true", help="xRFM 超参数网格搜索")

    # CV 模式
    parser.add_argument("--cv",         action="store_true", help="使用 GroupKFold CV 评估（推荐）")
    parser.add_argument("--n-splits",   type=int, default=5, help="CV 折数")

    # 数据路径
    parser.add_argument("--x-file",      default="data/仅自变量标准化数据.xlsx")
    parser.add_argument("--cluster-file",
        default="outputs/clustering/k3/tables/village_year_cluster_assignments.xlsx")
    parser.add_argument("--output-dir",  default="outputs/models")

    # 数据划分
    parser.add_argument("--val-size",  type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed",      type=int,   default=42)

    # xRFM 专用超参数
    parser.add_argument("--kernel",    default="l2_high_dim",
        choices=["sum_power_laplace", "l2_high_dim", "laplace"])
    parser.add_argument("--bandwidth", type=float, default=10.0)
    parser.add_argument("--reg",       type=float, default=0.1,
        help="xRFM 正则化系数，越大越正则（减少过拟合）")
    parser.add_argument("--iters",     type=int,   default=7)
    parser.add_argument("--early-stop-multiplier", type=float, default=1.02,
        help="xRFM 早停阈值，越小越激进（1.0=严格，1.1=宽松）")
    parser.add_argument("--max-leaf-size",  type=int, default=60_000)
    parser.add_argument("--n-trees",        type=int, default=1)
    parser.add_argument("--n-tree-iters",   type=int, default=0)
    parser.add_argument("--split-method",   default="top_vector_agop_on_subset",
        choices=["top_vector_agop_on_subset", "random_agop_on_subset",
                 "top_pc_agop_on_subset", "random_pca", "linear"])

    return parser.parse_args()


def save_cv_results(model_name: str, cv_result: dict, out_dir: Path,
                    feature_set: str = "full_80") -> None:
    """保存 CV 折详情和均值到 Excel。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cv_result:
        print(f"[输出] CV 结果为空，未保存 {model_name} 的结果。")
        return
    fold_df = pd.DataFrame(cv_result["fold_details"])
    summary_row = {
        "model":                  model_name,
        "feature_set":            feature_set,
        "train_accuracy_mean":    cv_result["train_accuracy"],
        "train_f1_macro_mean":    cv_result["train_f1_macro"],
        "test_accuracy_mean":     cv_result["accuracy"],
        "test_accuracy_std":      cv_result["accuracy_std"],
        "test_precision_macro_mean": cv_result["precision_macro"],
        "test_precision_macro_std":  cv_result["precision_std"],
        "test_recall_macro_mean": cv_result["recall_macro"],
        "test_recall_macro_std":  cv_result["recall_std"],
        "test_f1_macro_mean":     cv_result["f1_macro"],
        "test_f1_macro_std":      cv_result["f1_macro_std"],
        "test_auc_ovr_mean":      cv_result.get("auc_ovr"),
        "test_auc_ovr_std":       cv_result.get("auc_ovr_std"),
        "test_auc_ovo_mean":      cv_result.get("auc_ovo"),
        "test_auc_ovo_std":       cv_result.get("auc_ovo_std"),
    }
    path = out_dir / f"{model_name}_{feature_set}_cv_results.xlsx"
    with pd.ExcelWriter(path) as writer:
        fold_df.to_excel(writer, sheet_name="fold_details", index=False)
        pd.DataFrame([summary_row]).to_excel(writer, sheet_name="summary", index=False)
    print(f"[输出] CV 结果已保存至 {path}")


def save_cv_summary(all_results: dict[str, dict], out_dir: Path,
                    feature_set: str = "full_80") -> None:
    rows = []
    for model_name, r in all_results.items():
        rows.append({
            "model":                     model_name,
            "feature_set":               feature_set,
            "train_accuracy_mean":       r["train_accuracy"],
            "train_f1_macro_mean":       r["train_f1_macro"],
            "test_accuracy_mean":        r["accuracy"],
            "test_accuracy_std":         r["accuracy_std"],
            "test_precision_macro_mean": r["precision_macro"],
            "test_precision_macro_std":  r["precision_std"],
            "test_recall_macro_mean":    r["recall_macro"],
            "test_recall_macro_std":     r["recall_std"],
            "test_f1_macro_mean":        r["f1_macro"],
            "test_f1_macro_std":         r["f1_macro_std"],
            "test_auc_ovr_mean":         r.get("auc_ovr"),
            "test_auc_ovr_std":          r.get("auc_ovr_std"),
            "test_auc_ovo_mean":         r.get("auc_ovo"),
            "test_auc_ovo_std":          r.get("auc_ovo_std"),
        })
    summary = pd.DataFrame(rows).sort_values("test_f1_macro_mean", ascending=False).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"all_models_cv_summary_{feature_set}.xlsx"
    summary.to_excel(path, index=False)

    print("\n" + "=" * 90)
    print(f"模型对比汇总（GroupKFold CV，特征集 {feature_set}，按测试折 F1-macro 均值降序）")
    print("=" * 90)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[输出] CV 汇总结果已保存至 {path}")


def save_feature_set_comparison_summary(
    all_feature_set_results: dict[str, dict[str, dict]], out_dir: Path,
) -> None:
    """汇总不同特征集（14/73/80维）下的模型表现，便于横向对比。"""
    rows = []
    for feature_set, model_results in all_feature_set_results.items():
        for model_name, r in model_results.items():
            rows.append({
                "feature_set": feature_set,
                "model": model_name,
                "accuracy_mean": r["accuracy"],
                "accuracy_std": r["accuracy_std"],
                "f1_macro_mean": r["f1_macro"],
                "f1_macro_std": r["f1_macro_std"],
                "auc_ovr_mean": r.get("auc_ovr"),
                "auc_ovr_std": r.get("auc_ovr_std"),
                "auc_ovo_mean": r.get("auc_ovo"),
                "auc_ovo_std": r.get("auc_ovo_std"),
            })
    summary = pd.DataFrame(rows)
    if summary.empty:
        print("\n[提示] 没有可用的特征集结果，跳过特征集对比汇总（可能所有模型都被跳过）。")
        return
    col_order = ["feature_set", "model", "accuracy_mean", "accuracy_std",
                 "f1_macro_mean", "f1_macro_std",
                 "auc_ovr_mean", "auc_ovr_std", "auc_ovo_mean", "auc_ovo_std"]
    summary = summary[[c for c in col_order if c in summary.columns]]
    summary = summary.sort_values(["model", "feature_set"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_set_comparison_summary.xlsx"
    summary.to_excel(path, index=False)

    print("\n" + "=" * 100)
    print("特征集维度对比汇总（GroupKFold CV，相同 village 分组）")
    print("=" * 100)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))
    print(f"\n[输出] 特征集对比汇总已保存至 {path}")


def main() -> None:
    args = parse_args()

    # 超参数搜索优先处理，不走训练流程
    if args.tune_xrfm:
        tune_xrfm(args)
        return

    # 确定要运行的模型列表
    if args.all:
        models_to_run = ALL_MODELS
    else:
        models_to_run = [m for m in ALL_MODELS if getattr(args, m.replace("-", "_"), False)]

    if not models_to_run:
        print("请指定要训练的模型，例如：")
        print("  python scripts/main.py --xrfm")
        print("  python scripts/main.py --all --cv")
        print("使用 --help 查看所有选项。")
        return

    all_results: dict[str, dict] = {}

    if args.cv:
        # ── GroupKFold CV 模式 ──
        _, y, groups, df_full, feature_sets = prepare_data_cv(args)
        all_feature_set_results: dict[str, dict[str, dict]] = {}

        for feature_set, X in feature_sets.items():
            print(f"\n{'#'*70}")
            print(f"# 特征集: {feature_set}  (维度: {X.shape[1]})")
            print(f"{'#'*70}")
            feature_results: dict[str, dict] = {}

            for model_name in models_to_run:
                print(f"\n{'='*60}")
                print(f"  模型: {model_name.upper()}  [CV 模式, {args.n_splits} 折, 特征集 {feature_set}]")
                print(f"{'='*60}")
                cv_result = run_cv(model_name, X, y, groups, args,
                                   n_splits=args.n_splits, df=df_full)
                out_dir = PROJECT_ROOT / args.output_dir / model_name
                save_cv_results(model_name, cv_result, out_dir, feature_set=feature_set)

                # 模型整体被跳过（如 TabPFN/TabICL 未就绪）时 cv_result 为空，跳过后续处理
                if cv_result and "all_te_idx" in cv_result:
                    # 保存拼合后的测试预测（供混淆矩阵和 per-class 分析使用）
                    _save_cv_predictions(
                        model_name,
                        cv_result["all_te_idx"],
                        cv_result["all_y_true"],
                        cv_result["all_y_pred"],
                        df_full, out_dir, feature_set=feature_set,
                    )
                    feature_results[model_name] = cv_result
                    all_results[f"{model_name}_{feature_set}"] = cv_result
                else:
                    print(f"  [跳过] 模型 {model_name.upper()} 在特征集 {feature_set} 下无有效结果，略过。")

            all_feature_set_results[feature_set] = feature_results
            if len(feature_results) > 1:
                save_cv_summary(feature_results, PROJECT_ROOT / args.output_dir,
                                feature_set=feature_set)

        # 跨特征集汇总
        if len(feature_sets) > 1:
            save_feature_set_comparison_summary(
                all_feature_set_results, PROJECT_ROOT / args.output_dir,
            )

    else:
        # ── 固定 train/val/test 划分模式 ──
        X_train, y_train, X_val, y_val, X_test, y_test, test_df = prepare_data(args)

        for model_name in models_to_run:
            out_dir = PROJECT_ROOT / args.output_dir / model_name
            print(f"\n{'='*60}")
            print(f"  模型: {model_name.upper()}")
            print(f"{'='*60}")

            if model_name == "xrfm":
                metrics = run_xrfm(X_train, y_train, X_val, y_val, X_test, y_test,
                                    test_df, args)
            else:
                mod = __import__(f"models.{model_name}.model", fromlist=["run"])
                metrics = mod.run(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    test_df, out_dir, seed=args.seed,
                )

            if metrics:
                all_results[model_name] = metrics

        if len(all_results) > 1:
            save_summary(all_results, PROJECT_ROOT / args.output_dir)


if __name__ == "__main__":
    main()
