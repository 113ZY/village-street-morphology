from sklearn.linear_model import LogisticRegression
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
    print("\n[LR] 训练逻辑回归 …")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(X_train, y_train)

    train_metrics = evaluate(y_train, model.predict(X_train), "训练集", model.predict_proba(X_train))
    val_metrics   = evaluate(y_val,   model.predict(X_val),   "验证集", model.predict_proba(X_val))
    test_metrics  = evaluate(y_test,  model.predict(X_test),  "测试集", model.predict_proba(X_test))

    save_results(out_dir, "lr", train_metrics, val_metrics, test_metrics,
                 test_df, y_test, model.predict(X_test))
    return test_metrics
