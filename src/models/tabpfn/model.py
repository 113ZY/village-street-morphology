from tabpfn import TabPFNClassifier
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
    print("\n[TabPFN] 训练 TabPFN v2 …")
    try:
        model = TabPFNClassifier(
            n_estimators=8,
            random_state=seed,
            ignore_pretraining_limits=True,
        )
        # TabPFN 在 fit 时同时使用 train+val 效果更好（无需早停）
        X_fit = np.concatenate([X_train, X_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
        model.fit(X_fit, y_fit)
    except Exception as e:
        if "license" in str(e).lower() or "License" in str(e):
            print(f"[TabPFN] 跳过：需要先接受 TabPFN license。")
            print(f"  请在浏览器访问 https://ux.priorlabs.ai 登录并接受 license，")
            print(f"  然后在终端运行 `tabpfn login` 完成认证后重试。")
        else:
            print(f"[TabPFN] 跳过：{e}")
        return {}

    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)
    y_pred_test  = model.predict(X_test)

    train_metrics = evaluate(y_train, y_pred_train, "训练集", model.predict_proba(X_train))
    val_metrics   = evaluate(y_val,   y_pred_val,   "验证集", model.predict_proba(X_val))
    test_metrics  = evaluate(y_test,  y_pred_test,  "测试集", model.predict_proba(X_test))

    save_results(out_dir, "tabpfn", train_metrics, val_metrics, test_metrics,
                 test_df, y_test, y_pred_test)
    return test_metrics
