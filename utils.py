import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
from omegaconf import OmegaConf
import argparse
import math

def build_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args, unknown = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    return cfg

def standardize_fit(user_arr, rest_arr):
    U = user_arr.astype(np.float64)
    R = rest_arr.astype(np.float64)
    U_mean, U_std = U.mean(axis=0), U.std(axis=0) + 1e-6
    R_mean, R_std = R.mean(axis=0), R.std(axis=0) + 1e-6
    return U_mean, U_std, R_mean, R_std

def standardize_apply(uX, rX, U_mean, U_std, R_mean, R_std):
    uZ = (uX - U_mean) / U_std
    rZ = (rX - R_mean) / R_std
    return np.concatenate([uZ, rZ], axis=1)

class ClickIterable(IterableDataset):
    def __init__(self, files, user_arr, rest_arr, batch_size=128, shuffle_files=True):
        self.files = list(files)
        self.user_arr = user_arr
        self.rest_arr = rest_arr
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files

    def __iter__(self):
        files = self.files[:]
        if self.shuffle_files:
            rng = np.random.default_rng(7)
            rng.shuffle(files)
        for f in files:
            df = pd.read_parquet(f, columns=["user_id", "restaurant_id", "click"])
            df = df.sample(frac=1.0, random_state=13).reset_index(drop=True)

            u = df["user_id"].to_numpy()
            r = df["restaurant_id"].to_numpy()
            y = df["click"].astype(np.float32).to_numpy()

            uX = self.user_arr[u]   # (N, 30)
            rX = self.rest_arr[r]   # (N, 10)
            X = np.concatenate([uX, rX], axis=1)  # (N, 40)

            n = len(df)
            for i in range(0, n, self.batch_size):
                xb = X[i:i+self.batch_size]
                yb = y[i:i+self.batch_size]
                yield torch.from_numpy(xb).float(), torch.from_numpy(yb).float()

class TestIterable(IterableDataset):
    def __init__(self, files, user_arr, rest_arr, batch_size=1024):
        self.files = list(files)
        self.user_arr = user_arr
        self.rest_arr = rest_arr
        self.batch_size = batch_size
    def __iter__(self):
        for f in self.files:
            df = pd.read_parquet(f, columns=["user_id", "restaurant_id", "click"])
            u = df["user_id"].to_numpy()
            r = df["restaurant_id"].to_numpy()
            y = df["click"].astype(np.float32).to_numpy()
            uX = self.user_arr[u]
            rX = self.rest_arr[r]
            X = np.concatenate([uX, rX], axis=1)
            n = len(df)
            for i in range(0, n, self.batch_size):
                xb = X[i:i+self.batch_size]
                yb = y[i:i+self.batch_size]
                yield xb, yb

def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))