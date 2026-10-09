from sklearn.tree import DecisionTreeClassifier
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
    print("\n[DT] 训练决策树 …")
    # 用验证集选最优 max_depth
    best_depth, best_acc = None, -1
    for depth in [3, 5, 7, 10, None]:
        m = DecisionTreeClassifier(max_depth=depth, random_state=seed)
        m.fit(X_train, y_train)
        acc = (m.predict(X_val) == y_val).mean()
        if acc > best_acc:
            best_acc, best_depth = acc, depth

    model = DecisionTreeClassifier(max_depth=best_depth, random_state=seed)
    model.fit(X_train, y_train)
    print(f"  最优 max_depth={best_depth}，验证集 Accuracy={best_acc:.4f}")

    train_metrics = evaluate(y_train, model.predict(X_train), "训练集", model.predict_proba(X_train))
    val_metrics   = evaluate(y_val,   model.predict(X_val),   "验证集", model.predict_proba(X_val))
    test_metrics  = evaluate(y_test,  model.predict(X_test),  "测试集", model.predict_proba(X_test))

    save_results(out_dir, "dt", train_metrics, val_metrics, test_metrics,
                 test_df, y_test, model.predict(X_test))
    return test_metrics
