"""各模型共用的评估函数和结果保存逻辑。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score,
)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    y_proba: np.ndarray | None = None,
) -> dict:
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


def save_results(
    out_dir: Path,
    model_name: str,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([train_metrics, val_metrics, test_metrics])
    results.to_excel(out_dir / f"{model_name}_metrics.xlsx", index=False)
    print(f"\n[输出] 评估指标已保存至 {out_dir / f'{model_name}_metrics.xlsx'}")

    pred_df = pd.DataFrame({
        "VILLAGE_ID": test_df.sort_values(["VILLAGE_ID", "YEAR"])["VILLAGE_ID"].values,
        "YEAR":       test_df.sort_values(["VILLAGE_ID", "YEAR"])["YEAR"].values,
        "y_true": y_test,
        "y_pred": y_pred_test,
    })
    pred_df.to_excel(out_dir / f"{model_name}_test_predictions.xlsx", index=False)
    print(f"[输出] 测试集预测已保存至 {out_dir / f'{model_name}_test_predictions.xlsx'}")
