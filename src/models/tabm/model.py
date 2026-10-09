"""
TabM 分类器封装。
TabM 是纯 PyTorch 模块，这里手写训练循环并包装成与其他模型一致的接口。
输出形状：(batch, k, n_classes)，预测时对 k 维取均值后 argmax。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

import tabm
from models.base import evaluate, save_results


def _build_model(n_features: int, n_classes: int, k: int = 32) -> tabm.TabM:
    return tabm.TabM(
        n_num_features=n_features,
        d_out=n_classes,
        k=k,
        n_blocks=3,
        d_block=128,
        dropout=0.1,
        arch_type="tabm",
        start_scaling_init="random-signs",
    )


def _train(
    model: tabm.TabM,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 30,
    weight_decay: float = 1e-4,
) -> tabm.TabM:
    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).long().to(device)
    Xv = torch.from_numpy(X_val).float().to(device)
    yv = torch.from_numpy(y_val).long().to(device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            # out: (batch, k, n_classes) → mean over k → (batch, n_classes)
            out = model(xb).mean(dim=1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(Xv).mean(dim=1)
            val_loss = criterion(val_out, yv).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict(model: tabm.TabM, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).float().to(device)
        out = model(Xt).mean(dim=1)
        return out.argmax(dim=1).cpu().numpy()


def _predict_proba(model: tabm.TabM, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).float().to(device)
        out = model(Xt).mean(dim=1)
        proba = torch.softmax(out, dim=1)
        return proba.cpu().numpy()


def run(
    X_train, y_train, X_val, y_val, X_test, y_test,
    test_df: pd.DataFrame,
    out_dir: Path,
    seed: int = 42,
) -> dict:
    print("\n[TabM] 训练 TabM …")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = len(np.unique(y_train))
    # 标签转 0-based
    y_tr = y_train - 1
    y_v  = y_val   - 1

    model = _build_model(X_train.shape[1], n_classes)
    model = _train(model, X_train, y_tr, X_val, y_v, device)

    y_pred_train = _predict(model, X_train, device) + 1
    y_pred_val   = _predict(model, X_val,   device) + 1
    y_pred_test  = _predict(model, X_test,  device) + 1

    train_metrics = evaluate(y_train, y_pred_train, "训练集", _predict_proba(model, X_train, device))
    val_metrics   = evaluate(y_val,   y_pred_val,   "验证集", _predict_proba(model, X_val,   device))
    test_metrics  = evaluate(y_test,  y_pred_test,  "测试集", _predict_proba(model, X_test,  device))

    save_results(out_dir, "tabm", train_metrics, val_metrics, test_metrics,
                 test_df, y_test, y_pred_test)
    return test_metrics
