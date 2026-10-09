from tabicl import TabICLClassifier
from pathlib import Path
import numpy as np
import pandas as pd
from models.base import evaluate, save_results


def run(
    X_train, y_train, X_val, y_val, X_test, y_test,
    test_df: pd.DataFrame,
    out_dir: Path,
    seed: int = 42,
) -> dict:
    print("\n[TabICLv2] 训练 TabICL v2 …")
    try:
        X_fit = np.concatenate([X_train, X_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
        model = TabICLClassifier(n_estimators=8, random_state=seed)
        model.fit(X_fit, y_fit)
    except Exception as e:
        print(f"[TabICLv2] 跳过：{e}")
        return {}

    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)
    y_pred_test  = model.predict(X_test)

    train_metrics = evaluate(y_train, y_pred_train, "训练集", model.predict_proba(X_train))
    val_metrics   = evaluate(y_val,   y_pred_val,   "验证集", model.predict_proba(X_val))
    test_metrics  = evaluate(y_test,  y_pred_test,  "测试集", model.predict_proba(X_test))

    save_results(out_dir, "tabicl", train_metrics, val_metrics, test_metrics,
                 test_df, y_test, y_pred_test)
    return test_metrics
