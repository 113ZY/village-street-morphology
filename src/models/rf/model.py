from sklearn.ensemble import RandomForestClassifier
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
    print("\n[RF] 训练随机森林 …")
    # 用验证集选最优 n_estimators
    best_n, best_acc = 100, -1
    for n in [50, 100, 200, 300]:
        m = RandomForestClassifier(n_estimators=n, random_state=seed, n_jobs=-1)
        m.fit(X_train, y_train)
        acc = (m.predict(X_val) == y_val).mean()
        if acc > best_acc:
            best_acc, best_n = acc, n

    model = RandomForestClassifier(n_estimators=best_n, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"  最优 n_estimators={best_n}，验证集 Accuracy={best_acc:.4f}")

    train_metrics = evaluate(y_train, model.predict(X_train), "训练集", model.predict_proba(X_train))
    val_metrics   = evaluate(y_val,   model.predict(X_val),   "验证集", model.predict_proba(X_val))
    test_metrics  = evaluate(y_test,  model.predict(X_test),  "测试集", model.predict_proba(X_test))

    save_results(out_dir, "rf", train_metrics, val_metrics, test_metrics,
                 test_df, y_test, model.predict(X_test))
    return test_metrics
