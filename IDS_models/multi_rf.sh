#!/bin/sh -e

pip install --upgrade pip && pip install numpy pandas scipy scikit-learn optuna pyarrow neptune-client neptune-contrib plotly psutil seaborn

echo 'Start training' > /project/logs/two-stage/multi/randomforest/${GPULAB_JOB_ID}.log
cd /project/Two-Stage && python3 Multi-RF.py >> /project/logs/two-stage/multi/randomforest/${GPULAB_JOB_ID}.log 2>&1