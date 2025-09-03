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
import os
from omegaconf import OmegaConf
from torch.utils.data import IterableDataset, DataLoader
from utils import build_config, standardize_fit, standardize_apply, ClickIterable

# DATA_DIR = Path("data")
# BATCH_SIZE = 1024
# EPOCHS = 5

"""
○
Batch size: 1024 -> too big
○
Epochs: 5
○
Learning rate: 1E-8 -> too small to convergence
○
Maximum time per epoch: 45 seconds
○
Minimum accuracy: 0.7 -> too low
"""

INPUT_DIM = 30 + 10  # user + restaurant features

# DO NOT EDIT MODEL ARCHITECTURE
class RankNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)

def train():
    cfg = build_config()
    print(">>> Training config (OmegaConf):")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Torch threads cap
    torch.set_num_threads(int(cfg.cpu_threads))

    # Resolve paths
    DATA_DIR = cfg.data_dir
    TRAIN_GLOB = os.path.join(DATA_DIR, "train", "train_*.parquet") if "${" in cfg.train_glob else cfg.train_glob
    USER_FEAT = cfg.user_features
    REST_FEAT = cfg.restaurant_features
    MODEL_PATH = cfg.model_path
    SCALER_PATH = cfg.scaler_path

    # train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.parquet")))
    # user_features = pd.read_parquet(DATA_DIR / "user_features.parquet")
    # restaurant_features = pd.read_parquet(DATA_DIR / "restaurant_features.parquet")
    # train_df = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
    # train_df = train_df.merge(user_features, on=["user_id"]).merge(
    #     restaurant_features, on=["restaurant_id"]
    # )
    # num_batches = math.ceil(len(train_df) / BATCH_SIZE)

    # Load features (kept in memory; id tables are modest)
    uf = pd.read_parquet(USER_FEAT).sort_values("user_id").reset_index(drop=True)
    rf = pd.read_parquet(REST_FEAT).sort_values("restaurant_id").reset_index(drop=True)
    user_mat = uf[[f"f{i:02d}" for i in range(30)]].to_numpy(dtype=np.float32)
    rest_mat = rf[[f"f{i:02d}" for i in range(10)]].to_numpy(dtype=np.float32)

    # Fit scaler on id-level matrices
    U_mean, U_std, R_mean, R_std = standardize_fit(user_mat, rest_mat)

    device = torch.device("cpu")
    model = RankNet(INPUT_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.lr))

    train_files = glob.glob(TRAIN_GLOB)
    if not train_files:
        raise FileNotFoundError(f"No train parquet files found under {TRAIN_GLOB}")

    for epoch in tqdm(range(1, int(cfg.epochs) + 1), desc="Epoch Progress"):
        t0 = time.time()
        model.train()
        total_loss, total, correct = 0.0, 0, 0

        ds_iter = ClickIterable(
            train_files, user_mat, rest_mat,
            batch_size=int(cfg.batch_size),
            shuffle_files=bool(cfg.shuffle_files)
        )
        loader = DataLoader(ds_iter, batch_size=None)

        for xb, yb in tqdm(loader, desc="Loader Progress"):
            # apply scaling (split 30/10)
            uX = xb[:, :30].numpy()
            rX = xb[:, 30:].numpy()
            Xz = standardize_apply(uX, rX, U_mean, U_std, R_mean, R_std)
            xb = torch.from_numpy(Xz).float()

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.numel()
                total_loss += float(loss.item())

            if time.time() - t0 > float(cfg.max_epoch_sec):
                break

        acc = correct / max(1, total)
        avg_loss = total_loss / max(1, total // int(cfg.batch_size))
        elapsed = time.time() - t0
        print(f"Epoch {epoch}: loss={avg_loss:.4f} acc={acc:.4f} time={elapsed:.1f}s (cap {cfg.max_epoch_sec}s)")

    # Save artifacts
    model_scripted = torch.jit.script(model)
    model_scripted.save(MODEL_PATH)
    np.savez(SCALER_PATH, U_mean=U_mean, U_std=U_std, R_mean=R_mean, R_std=R_std)
    print(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")

    #     for batch in tqdm(range(num_batches), desc=f"epoch {epoch + 1}"):
    #         batch_df = train_df.iloc[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
    #         x = torch.tensor(
    #             batch_df.drop(
    #                 columns=[
    #                     "user_id",
    #                     "restaurant_id",
    #                     "click",
    #                     "latitude",
    #                     "longitude",
    #                 ]
    #             ).values,
    #             dtype=torch.float32,
    #         )
    #         y = torch.tensor(batch_df["click"].values.astype(np.float32))
    #         optimizer.zero_grad()
    #         output = model(x)
    #         loss = criterion(output, y)
    #         loss.backward()
    #         optimizer.step()
    #     print(f"Loss: {loss.item():.4f}, Elapsed time {time.time() - start_time:.3f} seconds")
    # model_scripted = torch.jit.script(model)
    # model_scripted.save(DATA_DIR / "model.pt")

if __name__ == "__main__":
    train()
