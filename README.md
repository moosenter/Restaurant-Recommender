# Resturant Recommender

Beginner-friendly steps to **generate data**, **train**, **evaluate**, and **serve** a restaurant recommender.

> Tested on Python 3.11, CPU-only. Works within **4 vCPU / 4GiB RAM** constraints when using default generator settings.

## Quick starts:
> This only works when `/data` dir exists in `data` must have the following files:
>
> model.pt, scaler.npz, user_features.parquet, restaurant_features.parquet, and docker are required priorly
```bash
chmod +x ./start_services.sh ./prepare.sh ./injection.sh
# First Terminal
./start_services.sh
# Second Terminal
./prepare.sh
# Third Terminal
./injection.sh
```

## 1) Setup

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

# How to

## 1. Generate data:
```python
python 01_generate_dataset.py
```
The data will be generated in `/data` foler. The generated data contains:
- train/train_*.parquet: Training dataset indicating if a user clicks a
restaurant.
- test/test_*.parquet: Test dataset indicating if a user clicks a restaurant.
- user_features.parquet: User preference features.
- restaurant_features.parquet: Restaurant characteristic features.
- requests.parquet: Generated API requests for performance testing.

## 2. Train & evaluate:
```python
python 02_train_model.py --config conf/train.yaml
python 03_evaluate.py --config conf/eval.yaml
```
The trained artifacts will be located in `/data` folder.

## 3. Serve & test:
Start Redis
```bash
docker run -p 6379:6379 --name redis -d redis:7-alpine
```
Load data to redis
```bash
python loader_redis.py
```
Serve and test
```python
export RECO_CONF=/conf/app.yaml
# python 04_inference.py
uvicorn 04_inference:app --host 0.0.0.0 --port 8123 --workers 2
# In another shell:
python perf_test/load_test.py --config conf/perf.yaml
# Or simple test run
curl -s -X POST http://localhost:8123/recommend/982159 -H "Content-Type: application/json" -d '{"candidate_restaurant_ids": [42457, 11352], "latitude": 13.7563, "longitude": 100.5018, "size": 20, "max_dist": 5000, "sort_dist": false}'
```
The expceted outputs should be:
```json
{
    "restaurants": [
        {
            "id": 42457,
            "score": 0.9974333643913269,
            "displacement": 4472.094431202431
        },
        {
            "id": 11352,
            "score": 0.02588445134460926,
            "displacement": 1112.6774468872095
        }
    ]
}
```

## 4. Docker deployments
build docker compose
```bash
docker-compose up --build
```
```bash
docker-compose run --rm loader
```
not build only run docker compose
```bash
docker-compose up
```
