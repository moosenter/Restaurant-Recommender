import __init__
import time, threading, queue, statistics, os, pathlib, argparse
import numpy as np
import pandas as pd
import requests
from omegaconf import OmegaConf
from utils import build_config

def load_requests(cfg):
    p = pathlib.Path(cfg.requests_file)
    df = pd.read_parquet(p)
    print(f'load rows: {len(df)}')
    return df

def worker(q: queue.Queue, results: list, url: str):
    s = requests.Session()
    while True:
        item = q.get()
        if item is None:
            break
        uid, body = item
        t0 = time.time()
        try:
            r = s.post(url.format(uid=uid), json=body, timeout=10)
            r.raise_for_status()
        except Exception:
            # record as 10s timeout to reflect failure
            results.append(10_000.0)
            q.task_done()
            continue
        latency = (time.time() - t0) * 1000.0
        results.append(latency)
        q.task_done()

def main():
    cfg = build_config()
    print(">>> Perf config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    url = cfg.api_base + cfg.endpoint
    df = load_requests(cfg)

    q = queue.Queue(maxsize=int(cfg.rps)*2)
    results = []
    threads = [threading.Thread(target=worker, args=(q, results, url), daemon=True) for _ in range(int(cfg.threads))]
    for t in threads:
        t.start()

    t_end = time.time() + int(cfg.duration_sec)
    i = 0
    sec = 0

    while time.time() < t_end:
        start = time.time()
        for _ in range(int(cfg.rps)):
            row = df.iloc[i % len(df)]
            body = {
                "candidate_restaurant_ids": list(map(int, row["candidate_restaurant_ids"])),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "size": int(row.get("size", int(cfg.top_k))),
                "max_dist": int(row.get("max_dist", int(cfg.max_dist))),
                "sort_dist": bool(row.get("sort_dist", bool(cfg.sort_dist))),
            }
            q.put((int(row["user_id"]), body))
            i += 1
        elapsed = time.time() - start
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        sec += 1
        print(f'{sec} sec passed - {int(cfg.rps)*sec} req passed')

    q.join()
    for _ in threads: 
        q.put(None)
    for t in threads: 
        t.join(timeout=1)

    if results:
        results = sorted(results)
        p50 = statistics.median(results)
        try:
            p95 = statistics.quantiles(results, n=100)[94]
            p99 = statistics.quantiles(results, n=100)[98]
        except Exception:
            p95 = p99 = float('nan')
        print(f"Requests: {len(results)} | p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms")
        
        out_dir = pathlib.Path("perf_test")
        out_dir.mkdir(exist_ok=True)

        with open(out_dir / "summary.txt", "w") as f:
            f.write(f"Requests: {len(results)} | p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms\\n")
        print("Saved perf_test/summary.txt")
    else:
        print("No results recorded. Is the API running?")
    

if __name__ == "__main__":
    main()