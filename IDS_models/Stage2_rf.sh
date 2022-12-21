#!/bin/sh -e

pip install --upgrade pip && pip install numpy pandas scipy scikit-learn optuna pyarrow neptune-client plotly psutil seaborn
bash /project/install_mysql_no_sudo.sh

mkdir -p /project/logs/two-stage/stage2/rf
echo 'Start training' > /project/logs/two-stage/stage2/rf/${GPULAB_JOB_ID}.log
cd /project/Two-Stage && python3 Stage2_RandomForest.py >> /project/logs/two-stage/stage2/rf/${GPULAB_JOB_ID}.log 2>&1