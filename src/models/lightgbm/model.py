import lightgbm as lgb
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
    print("\n[LightGBM] 训练 LightGBM …")
    # 标签从 1-based 转为 0-based
    y_tr = y_train - 1
    y_v  = y_val   - 1
    y_te = y_test  - 1

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        X_train, y_tr,
        eval_set=[(X_val, y_v)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)],
    )

    y_pred_train = model.predict(X_train) + 1
    y_pred_val   = model.predict(X_val)   + 1
    y_pred_test  = model.predict(X_test)  + 1

    train_metrics = evaluate(y_train, y_pred_train, "训练集", model.predict_proba(X_train))
    val_metrics   = evaluate(y_val,   y_pred_val,   "验证集", model.predict_proba(X_val))
    test_metrics  = evaluate(y_test,  y_pred_test,  "测试集", model.predict_proba(X_test))

    save_results(out_dir, "lightgbm", train_metrics, val_metrics, test_metrics,
                 test_df, y_test, y_pred_test)
    return test_metrics
