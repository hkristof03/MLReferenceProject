### start mlflow UI:
mlflow ui --backend-store-uri ./train/artifacts/mlflow
http://localhost:5000/

### tensorboard
tensorboard --logdir train/artifacts/logs
http://localhost:6006/

### monitor resource usage
nvitop

### start fastapi app
uvicorn app:app --host 0.0.0.0 --port 8000

### Build the docker images:
- docker build -t ml_base -f ./Dockerfile .
- docker build -t inference -f ./inference/Dockerfile .

### Start with Docker compose:
docker compose up -d