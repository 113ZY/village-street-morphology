import xgboost as xgb
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
    print("\n[XGBoost] 训练 XGBoost …")
    # 标签从 1-based 转为 0-based（XGBoost 要求）
    y_tr = y_train - 1
    y_v  = y_val   - 1
    y_te = y_test  - 1

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=seed,
        verbosity=0,
    )
    model.fit(
        X_train, y_tr,
        eval_set=[(X_val, y_v)],
        verbose=False,
    )

    y_pred_train = model.predict(X_train) + 1
    y_pred_val   = model.predict(X_val)   + 1
    y_pred_test  = model.predict(X_test)  + 1

    train_metrics = evaluate(y_train, y_pred_train, "训练集", model.predict_proba(X_train))
    val_metrics   = evaluate(y_val,   y_pred_val,   "验证集", model.predict_proba(X_val))
    test_metrics  = evaluate(y_test,  y_pred_test,  "测试集", model.predict_proba(X_test))

    save_results(out_dir, "xgboost", train_metrics, val_metrics, test_metrics,
                 test_df, y_test, y_pred_test)
    return test_metrics
