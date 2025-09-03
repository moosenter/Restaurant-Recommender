import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
from omegaconf import OmegaConf
import pathlib, json, math, time
from utils import haversine_vec, build_config, standardize_apply
from dotenv import load_dotenv
import redis

load_dotenv()

_here = pathlib.Path(__file__).resolve().parent
_default_cfg = _here / "conf" / "app.yaml"
_cfg_path = os.environ.get("APPCONF", str(_default_cfg))
cfg = OmegaConf.load(_cfg_path)
cfg = OmegaConf.to_container(cfg, resolve=True)  # plain dict for speed

DATA_DIR = cfg["data_dir"]
USER_FEAT = cfg["paths"]["user_features"]
REST_FEAT = cfg["paths"]["restaurant_features"]
MODEL_PATH = cfg["paths"]["model_path"]
SCALER_PATH = cfg["paths"]["scaler_path"]
DB_PATH = cfg["paths"]["db_path"]

print(">>> APP config:")
print(OmegaConf.to_yaml(cfg, resolve=True))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class RecommendBody(BaseModel):
    candidate_restaurant_ids: List[int]
    latitude: float
    longitude: float
    size: int = 20
    max_dist: int = 5000
    sort_dist: bool = False

model: Optional[torch.nn.Module] = None
U_mean = U_std = R_mean = R_std = None

R_CFG = cfg["db"]["redis"]
REDIS_HOST = R_CFG.get("host", "127.0.0.1")
REDIS_PORT = int(R_CFG.get("port", 6379))
REDIS_DB   = int(R_CFG.get("db", 0))
K_U = R_CFG.get("prefix_users", "u:")
K_R = R_CFG.get("prefix_restaurants", "r:")
rds = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False, socket_timeout=0.5, socket_connect_timeout=0.5)

# def load_data_to_redis():
#     user_parquet = cfg["paths"]["user_features"]
#     rest_parquet = cfg["paths"]["restaurant_features"]

#     # Users
#     dfu = pd.read_parquet(user_parquet).sort_values("user_id")
#     uf = dfu[[f"f{i:02d}" for i in range(30)]].to_numpy(dtype=np.float32)
#     uid = dfu["user_id"].to_numpy(dtype=np.int64)
#     pipe = rds.pipeline(transaction=False)
#     batch = int(R_CFG.get("batch", 20000))
#     for i in range(len(dfu)):
#         key = (R_CFG.get("prefix_users","u:") + str(int(uid[i]))).encode()
#         pipe.set(key, uf[i].tobytes())
#         if (i+1) % batch == 0:
#             pipe.execute()
#     pipe.execute()

#     # Restaurants
#     dfr = pd.read_parquet(rest_parquet).sort_values("restaurant_id")
#     rf = dfr[[f"f{i:02d}" for i in range(10)]].to_numpy(dtype=np.float32)
#     rid = dfr["restaurant_id"].to_numpy(dtype=np.int64)
#     pipe = rds.pipeline(transaction=False)
#     for i in range(len(dfr)):
#         key = (R_CFG.get("prefix_restaurants","r:") + str(int(rid[i]))).encode()
#         pipe.set(key, rf[i].tobytes())
#         if (i+1) % batch == 0:
#             pipe.execute()
#     pipe.execute()

#     print(f"Loaded {len(dfu)} users and {len(dfr)} restaurants into Redis.")

scal = np.load(cfg['paths']['scaler_path'])
U_mean, U_std = scal["U_mean"], scal["U_std"]
R_mean, R_std = scal["R_mean"], scal["R_std"]

model = torch.jit.load(cfg['paths']['model_path'])
model.eval()
example = torch.zeros(1, 40, dtype=torch.float32)
model = torch.jit.trace(model, example)
model = torch.jit.freeze(model)

# load_data_to_redis()

def _get_user_vector(uid: int) -> np.ndarray:
    key = (K_U + str(int(uid))).encode()
    b = rds.get(key)
    if b is None:
        raise KeyError(f"user {uid} not in Redis")
    return np.frombuffer(b, dtype=np.float32, count=30)

def _get_rest_vectors(rids: np.ndarray) -> np.ndarray:
    keys = [(K_R + str(int(r))).encode() for r in rids.tolist()]
    vals = rds.mget(keys)
    # Replace missing with zeros to avoid failures
    out = np.zeros((len(keys), 10), dtype=np.float32)
    for i, v in enumerate(vals):
        if v is not None:
            out[i] = np.frombuffer(v, dtype=np.float32, count=10)
    return out

app = FastAPI(title="Recommender API", version="1.0.0")

@app.get("/ping")
def ping():
    return {"pong": True, "time": time.time()}

@app.get("/healthz")
def healthz():
    status = {
        "model_file_exists": os.path.exists(MODEL_PATH),
        "scaler_file_exists": os.path.exists(SCALER_PATH),
        "db_exists": os.path.exists(DB_PATH),
        "user_features_exists": os.path.exists(USER_FEAT),
        "restaurant_features_exists": os.path.exists(REST_FEAT),
        "model_loaded": model is not None,
        "scaler_loaded": U_mean is not None and U_std is not None and R_mean is not None and R_std is not None,
        "status": "ok" if (os.path.exists(USER_FEAT) and os.path.exists(REST_FEAT)) else "degraded",
        "config_path": _cfg_path
    }
    return status

@app.post("/echo")
async def echo(req: Request):
    body = await req.json()
    return {"ok": True, "received": body, "time": time.time()}

@app.post("/recommend/{user_id}")
def recommend(user_id: int, body: RecommendBody):
    results = []
    if model is None or rds is None:
        return {"error": "server not ready"}
    
    cand = np.asarray(body.candidate_restaurant_ids, dtype=np.int64)
    if cand.size == 0:
        return {"restaurants": []}
    
    try:
        u = _get_user_vector(int(user_id))
    except KeyError:
        # fallback: cold miss -> zeros
        u = np.zeros((30,), dtype=np.float32)

    R = _get_rest_vectors(cand)

    R_feats = R
    U_feats = np.repeat(u.reshape(1, -1), len(cand), axis=0)

    if any(x is None for x in [U_mean, U_std, R_mean, R_std]):
        return {"error": "scaler not loaded; missing data/scaler.npz. Run training first."}
    
    X = standardize_apply(U_feats, R_feats, U_mean, U_std, R_mean, R_std)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
        scores = torch.sigmoid(logits).numpy().tolist()

    lat = float(body.latitude)
    lon = float(body.longitude)
    lat2 = lat + ((cand % 100) - 50) * 1e-4
    lon2 = lon + ((cand // 100) - 10) * 1e-4
    disps = haversine_vec(lat, lon, lat2, lon2)

    for rid, score, disp in zip(cand, scores, disps):
        if disp <= body.max_dist:
            results.append({"id": int(rid), "score": float(score), "displacement": float(disp)})

    if body.sort_dist:
        results.sort(key=lambda r: r["displacement"])
    else:
        results.sort(key=lambda r: r["score"], reverse=True)

    # print(
    #     json.dumps(
    #         {"restaurants" : results[:body.size]},
    #         indent=4,
    #         cls=NpEncoder,
    #     )
    # )

    return {"restaurants": results[:body.size]}

# if __name__ == "__main__":
    # uvicorn.run("04_inference:app", host=cfg["api"]["host"], port=int(cfg["api"]["port"]), workers=1)


# request_df = pd.read_parquet(DATA_DIR / "requests.parquet")
# user_df = pd.read_parquet(DATA_DIR / "user_features.parquet")
# restaurant_df = pd.read_parquet(DATA_DIR / "restaurant_features.parquet").set_index(
#     "restaurant_id", drop=True
# )

# model = torch.jit.load(DATA_DIR / "model.pt")
# model.eval()

# request = request_df.iloc[0]
# user_features = (
#     user_df[user_df["user_id"] == request["user_id"]].drop(columns="user_id").values
# )
# restaurant_features = (
#     restaurant_df.loc[request["candidate_restaurant_ids"]]
#     .drop(columns=["latitude", "longitude"])
#     .values
# )
# x = torch.tensor(
#     np.hstack(
#         (
#             np.tile(user_features, (len(request["candidate_restaurant_ids"]), 1)),
#             restaurant_features,
#         )
#     ),
#     dtype=torch.float32,
# )

# with torch.no_grad():
#     y_pred = torch.sigmoid(model(x))

# result = [
#     {"restaurant_id": rid, "score": prob}
#     for rid, prob in zip(
#         request["candidate_restaurant_ids"],
#         y_pred.numpy(),
#     )
# ]
# sorted_result = sorted(result, key=lambda item: item["score"], reverse=True)
# print(
#     json.dumps(
#         sorted_result,
#         indent=4,
#         cls=NpEncoder,
#     )
# )
