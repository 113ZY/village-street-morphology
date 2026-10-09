"""汇总各模型 × 各特征集的 CV 结果到一份完整对比表。

读取 outputs/models/<model>/<model>_<feature_set>_cv_results.xlsx 的 summary 页，
合并生成：
  - outputs/models/feature_set_comparison_summary.xlsx  (横向：14/73/80 维对比)
  - outputs/models/all_models_cv_summary_<feature_set>.xlsx  (每个特征集下所有模型)
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "outputs" / "models"
FEATURE_SETS = ["raw_14", "dynamic_73", "full_80"]
MODELS = ["lr", "dt", "rf", "xgboost", "lightgbm", "tabm", "tabpfn", "tabicl", "xrfm"]

# 需要列（summary 页里的字段）
NEED = [
    "model", "feature_set",
    "train_accuracy_mean", "train_f1_macro_mean",
    "test_accuracy_mean", "test_accuracy_std",
    "test_precision_macro_mean", "test_precision_macro_std",
    "test_recall_macro_mean", "test_recall_macro_std",
    "test_f1_macro_mean", "test_f1_macro_std",
    "test_auc_ovr_mean", "test_auc_ovr_std",
    "test_auc_ovo_mean", "test_auc_ovo_std",
]

rows = []
missing = []
for model in MODELS:
    for fs in FEATURE_SETS:
        f = MODELS_DIR / model / f"{model}_{fs}_cv_results.xlsx"
        if not f.exists():
            missing.append(f"{model}/{fs}")
            continue
        try:
            summ = pd.read_excel(f, sheet_name="summary")
            row = summ.iloc[0].to_dict()
            rows.append({k: row.get(k) for k in NEED})
        except Exception as e:
            missing.append(f"{model}/{fs}: {e}")

if not rows:
    raise SystemExit("没有找到任何 CV 结果文件。")

combined = pd.DataFrame(rows)

# 1) 特征集维度对比（按 model+feature_set 排序）
comp = combined.sort_values(["model", "feature_set"]).reset_index(drop=True)
comp_path = MODELS_DIR / "feature_set_comparison_summary.xlsx"
comp.to_excel(comp_path, index=False)
print(f"[输出] 特征集对比汇总 -> {comp_path}")
print(comp[["feature_set", "model", "test_accuracy_mean", "test_f1_macro_mean",
           "test_auc_ovr_mean", "test_auc_ovo_mean"]].to_string(index=False,
           float_format=lambda x: f"{x:.4f}"))

# 2) 每个特征集下所有模型的汇总
for fs in FEATURE_SETS:
    sub = combined[combined["feature_set"] == fs].copy()
    if sub.empty:
        continue
    sub = sub.sort_values("test_f1_macro_mean", ascending=False).reset_index(drop=True)
    p = MODELS_DIR / f"all_models_cv_summary_{fs}.xlsx"
    sub.to_excel(p, index=False)
    print(f"\n[输出] {fs} 模型汇总 -> {p}")

print("\n缺失（未跑/无结果）:", missing if missing else "无")
