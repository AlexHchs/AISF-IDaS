#!/bin/sh -e

pip install --upgrade pip && pip install numpy pandas scipy scikit-learn optuna pyarrow neptune-client neptune-contrib plotly psutil

echo 'Start training' > /project/logs/two-stage/binary/ocsvm/${GPULAB_JOB_ID}.log
cd /project/Two-Stage && python3 ocsvm.py >> /project/logs/two-stage/binary/ocsvm/${GPULAB_JOB_ID}.log 2>&1