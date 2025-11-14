# Quick run

1. Train model:
   python train.py --data-dir ./data --out models/model_bundle.joblib

2. Run service locally (after training/model exists):
   python service/app.py

3. Or build docker image and run:
   docker build -t zonal-predictor .
   docker run -p 5000:5000 zonal-predictor

POST /predict accepts multipart/form-data with `file` field or raw CSV body. It returns JSON `{predicted_zone: "<zone>"}`.
