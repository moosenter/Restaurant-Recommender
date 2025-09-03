from typing import List, Optional
from omegaconf import OmegaConf
import pathlib, os, json, math, time
from utils import haversine_vec, build_config, standardize_apply
from dotenv import load_dotenv
import redis
import pandas as pd
import numpy as np
from tqdm import tqdm

load_dotenv()

_here = pathlib.Path(__file__).resolve().parent
_default_cfg = _here / "conf" / "app.yaml"
_cfg_path = os.environ.get("APPCONF", str(_default_cfg))
cfg = OmegaConf.load(_cfg_path)
cfg = OmegaConf.to_container(cfg, resolve=True)  # plain dict for speed

R_CFG = cfg["db"]["redis"]
REDIS_HOST = R_CFG.get("host", "127.0.0.1")
REDIS_PORT = int(R_CFG.get("port", 6379))
REDIS_DB   = int(R_CFG.get("db", 0))
K_U = R_CFG.get("prefix_users", "u:")
K_R = R_CFG.get("prefix_restaurants", "r:")
rds = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False, socket_timeout=0.5, socket_connect_timeout=0.5)

def load_data_to_redis():
    user_parquet = cfg["paths"]["user_features"]
    rest_parquet = cfg["paths"]["restaurant_features"]

    # Users
    dfu = pd.read_parquet(user_parquet).sort_values("user_id")
    uf = dfu[[f"f{i:02d}" for i in range(30)]].to_numpy(dtype=np.float32)
    uid = dfu["user_id"].to_numpy(dtype=np.int64)
    pipe = rds.pipeline(transaction=False)
    batch = int(R_CFG.get("batch", 20000))
    for i in tqdm(range(len(dfu))):
        key = (R_CFG.get("prefix_users","u:") + str(int(uid[i]))).encode()
        pipe.set(key, uf[i].tobytes())
        if (i+1) % batch == 0:
            pipe.execute()
    pipe.execute()

    # Restaurants
    dfr = pd.read_parquet(rest_parquet).sort_values("restaurant_id")
    rf = dfr[[f"f{i:02d}" for i in range(10)]].to_numpy(dtype=np.float32)
    rid = dfr["restaurant_id"].to_numpy(dtype=np.int64)
    pipe = rds.pipeline(transaction=False)
    for i in tqdm(range(len(dfr))):
        key = (R_CFG.get("prefix_restaurants","r:") + str(int(rid[i]))).encode()
        pipe.set(key, rf[i].tobytes())
        if (i+1) % batch == 0:
            pipe.execute()
    pipe.execute()

    print(f"Loaded {len(dfu)} users and {len(dfr)} restaurants into Redis.")

if __name__ == "__main__":
    load_data_to_redis()