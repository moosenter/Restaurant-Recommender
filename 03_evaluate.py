# train_model.py
import gc
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse, os
from torch.utils.data import IterableDataset, DataLoader
from utils import build_config, standardize_apply, TestIterable

def evaluate():

    cfg = build_config()
    print(">>> Eval config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    uf = pd.read_parquet(cfg.user_features).sort_values("user_id").reset_index(drop=True)
    rf = pd.read_parquet(cfg.restaurant_features).sort_values("restaurant_id").reset_index(drop=True)
    user_mat = uf[[f"f{i:02d}" for i in range(30)]].to_numpy(dtype=np.float32)
    rest_mat = rf[[f"f{i:02d}" for i in range(10)]].to_numpy(dtype=np.float32)

    scal = np.load(cfg.scaler_path)
    U_mean, U_std = scal["U_mean"], scal["U_std"]
    R_mean, R_std = scal["R_mean"], scal["R_std"]

    model = torch.jit.load(cfg.model_path)
    model.eval()

    files = glob.glob(cfg.test_glob)
    if not files:
        raise FileNotFoundError(f"No test files matching {cfg.test_glob}")
    
    ds = TestIterable(files, user_mat, rest_mat, batch_size=int(cfg.batch_size))
    loader = DataLoader(ds, batch_size=None)

    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Loader Progress"):
            uX, rX = xb[:, :30], xb[:, 30:]
            Xz = standardize_apply(uX, rX, U_mean, U_std, R_mean, R_std)
            logits = model(torch.from_numpy(Xz).float())
            prob = torch.sigmoid(logits).numpy()
            y_true.extend(yb.tolist())
            y_prob.extend(prob.tolist())

    y_pred = [1.0 if p >= 0.5 else 0.0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    print(f"Test Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

    # test_files = sorted(glob.glob(str(DATA_DIR / "test" / "*.parquet")))
    # user_features = pd.read_parquet(DATA_DIR / "user_features.parquet")
    # restaurant_features = pd.read_parquet(DATA_DIR / "restaurant_features.parquet")

    # model = torch.jit.load(DATA_DIR / "model.pt")

    # test_df = pd.concat([pd.read_parquet(f) for f in test_files], ignore_index=True)
    # test_df = test_df.merge(user_features, on=["user_id"]).merge(
    #     restaurant_features, on=["restaurant_id"]
    # )
    # model.eval()
    # with torch.no_grad():
    #     x = torch.tensor(
    #         test_df.drop(
    #             columns=[
    #                 "user_id",
    #                 "restaurant_id",
    #                 "click",
    #                 "latitude",
    #                 "longitude",
    #             ]
    #         ).values,
    #         dtype=torch.float32,
    #     )
    #     y = torch.tensor(test_df["click"].values.astype(np.float32))
    #     output = torch.sigmoid(model(x))
    #     pred = (output > 0.5).float()
    #     correct = (pred == y).sum().item()
    #     total = y.size(0)
    # print(f"Test Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    evaluate()
