# Seed value
# Apparently you may use different seed values at each stage
seed_value= 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

import pandas as pd
import util.common as util
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.metrics import auc, roc_curve, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pathlib
import pickle
import matplotlib.pyplot as plt

# from keras.models import Model
# from keras.layers import Dense, Input
# from keras.regularizers import l2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l2
from util.AUROCEarlyStoppingPruneCallback import AUROCEarlyStoppingPruneCallback

import neptune.new as neptune
from neptune.new.types import File

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import ThresholdPruner
import optuna.visualization as vis

train = {
    "ocsvm": {}, # 10k samples
    "ae": {}, # 100k samples
    "stage2": {}
}
val = {
    "ocsvm": {},
    "ae": {},
    "stage2": {}
}
test = {
    # "y"
    # "y_binary"
    # "y_unknown"
    # "x"
}

# Load Data Stage 1
clean_dir = "/project/data/cicids2017/clean/"


train["ocsvm"]["x"], train["ocsvm"]["y"], x_benign_val, y_benign_val, _, _, x_malicious_train, y_malicious_train, _, _, _, _, _ = util.load_data(clean_dir, sample_size=1948, train_size=10000, val_size=129485, test_size=56468)

val["ocsvm"]["x"] = np.concatenate((x_benign_val, x_malicious_train))
val["ocsvm"]["y"] = np.concatenate((y_benign_val, np.full(y_malicious_train.shape[0], -1)))


train["ae"]["x"], train["ae"]["y"], x_benign_val, y_benign_val, _, _, x_malicious_train, y_malicious_train, _, _, _, _, _ = util.load_data(clean_dir, sample_size=1948, val_size=129485, test_size=56468)

val["ae"]["x"] = np.concatenate((x_benign_val, x_malicious_train))
val["ae"]["y"] = np.concatenate((y_benign_val, np.full(y_malicious_train.shape[0], -1)))

# Load Data Stage 2
n_benign_val = 1500

x_benign_train, _, _, _, x_benign_test, y_benign_test, x_malicious_train, y_malicious_train, x_malicious_test, y_malicious_test, attack_type_train, _, _ = util.load_data(clean_dir, sample_size=1948, train_size=n_benign_val, val_size=6815, test_size=56468)
train["stage2"]["x"], x_val, train["stage2"]["y"], y_val = train_test_split(x_malicious_train, y_malicious_train, stratify=attack_type_train, test_size=1500, random_state=42, shuffle=True)

test['x'] = np.concatenate((x_benign_test, x_malicious_test))
test["y_n"] = np.concatenate((y_benign_test, np.full(y_malicious_test.shape[0], -1)))

val["stage2"]["x"] = np.concatenate((x_val, x_benign_train))
val["stage2"]["y"] = np.concatenate((y_val, np.full(n_benign_val, "Unknown")))

train["stage2"]["y_n"] = pd.get_dummies(train["stage2"]["y"])
val["stage2"]["y_n"] = pd.get_dummies(val["stage2"]["y"])

test["y"] = np.concatenate((np.full(56468, "Benign"), y_malicious_test))
test["y_unknown"] = np.where((test["y"] == "Heartbleed") | (test["y"] == "Infiltration"), "Unknown", test["y"])
test["y_unknown_all"] = np.where(test['y_unknown'] == 'Benign', "Unknown", test['y_unknown'])

# Scale the data

scaler = QuantileTransformer(output_distribution='normal')
train['ocsvm']['x_s'] = scaler.fit_transform(train['ocsvm']['x'])
val['ocsvm']['x_s'] = scaler.transform(val['ocsvm']['x'])
test['ocsvm_s'] = scaler.transform(test['x'])

scaler = QuantileTransformer(output_distribution='normal')
train['ae']['x_s'] = scaler.fit_transform(train['ae']['x'])
val['ae']['x_s'] = scaler.transform(val['ae']['x'])
test['ae_s'] = scaler.transform(test['x'])

scaler = QuantileTransformer(output_distribution='normal')
train['stage2']['x_s'] = scaler.fit_transform(train['stage2']['x'])
val['stage2']['x_s'] = scaler.transform(val['stage2']['x'])
test['stage2_s'] = scaler.transform(test['x'])

scaler = QuantileTransformer(output_distribution='uniform')
train['stage2']['x_q'] = scaler.fit_transform(train['stage2']['x'])
val['stage2']['x_q'] = scaler.transform(val['stage2']['x'])
test['stage2_q'] = scaler.transform(test['x'])

predictions = {
    "stage1": [],
    "stage2": [],
    "y_unk": test["y_unknown"]
}

thresholds = {
    "stage1": [],
    "stage2": [],
    "extension": []
}

quantiles = [0.995, 0.99, 0.975, 0.95]

# Stage 1
best_runs_1 = pd.read_csv('results/stage1.csv')
val_results_1 = []
test_results_1 = []

# Iterate best models in stage 1
for index, row in best_runs_1.iterrows():
    print(f"Stage 1: {index}")
    # Predict anomaly score on validation (sanity check) and test set
    if row["algo"] == 'ae':
        model_1 = load_model(f'results/stage1/{row["algo"]}/{row["run_id"]}/models/model_{row["number"]}.h5')
        x_pred_val = model_1.predict(val['ae']['x_s'])
        score_val = util.anomaly_scores(val['ae']['x_s'], x_pred_val)
        curves_metrics, summary_metrics = util.evaluate_proba(val['ae']['y'], score_val)
        thresholds["extension"].append({q: np.quantile(score_val[val["ae"]["y"] == 1], q) for q in quantiles})

        x_pred_test = model_1.predict(test['ae_s'])
        score_test = util.anomaly_scores(test['ae_s'], x_pred_test)
    else:
        f = open(f'results/stage1/{row["algo"]}/{row["run_id"]}/models/model_{row["number"]}.p', 'rb')
        model_1 = pickle.load(f)
        f.close()
        score_val = -model_1.decision_function(val['ocsvm']['x_s'])
        curves_metrics, summary_metrics = util.evaluate_proba(val['ocsvm']['y'], score_val)
        thresholds["extension"].append({q: np.quantile(score_val[val["ocsvm"]["y"] == 1], q) for q in quantiles})

        score_test = -model_1.decision_function(test['ocsvm_s'])
    # Save predictions to dictionary
    predictions['stage1'].append(score_test)
    thresholds["stage1"].append({(metric, fpr): t for metric, fpr, t in zip(summary_metrics.metric, summary_metrics.FPR, summary_metrics.threshold)})

    
    # Collect validation metrics
    val_metric = {
        "algo": row["algo"],
        "run_id": row["run_id"],
        "trial_id": row["number"]
    }
    test_metric = val_metric.copy()
    for i, row in summary_metrics.iterrows():
        val_metric[row['metric']] = row['value']
    fpr, tpr, t = roc_curve(val['ae']['y'], score_val, pos_label=-1)
    val_metric["AUROC"] = auc(fpr, tpr)
    val_metric["au_precision_recall"] = auc(curves_metrics['recall'], curves_metrics['precision'])
    val_results_1.append(val_metric)

    curves_metrics_test, summary_metrics_test = util.evaluate_proba(test["y_n"], score_test)
    for i, row in summary_metrics_test.iterrows():
        test_metric[row['metric']] = row['value']
    fpr, tpr, t = roc_curve(test["y_n"], score_test, pos_label=-1)
    test_metric["AUROC"] = auc(fpr, tpr)
    test_metric["au_precision_recall"] = auc(curves_metrics_test['recall'], curves_metrics_test['precision'])
    test_results_1.append(test_metric)
    
# Save validation metrics stage 1 on disk
df = pd.DataFrame(val_results_1)
df.to_csv("results/val_stage1.csv", index=False)
# Save test metrics stage 1 on disk
df = pd.DataFrame(test_results_1)
df.to_csv("results/test_stage1.csv", index=False)
        
# Stage 2
best_runs_2 = pd.read_csv('results/stage2.csv')
val_results_2 = []
test_results_2 = []

# Iterate best models in stage 2
for index, row in best_runs_2.iterrows():
    print(f"Stage 2: {index}")
    # Predict class probability on validation (sanity check) and test set
    if row['algo'] == "nn":
        model_2 = load_model(f'results/stage2/{row["algo"]}/{row["run_id"]}/models/model_{row["number"]}.h5')
        if row['scaler'] == 'normal':
            y_proba_val_2 = model_2.predict(val['stage2']['x_s'])
            y_proba_test_2 = model_2.predict(test['stage2_s']) 
        else: # 'uniform'
            y_proba_val_2 = model_2.predict(val['stage2']['x_q'])
            y_proba_test_2 = model_2.predict(test['stage2_q'])
        fmacro, fweight, threshold_fscore, f_best = util.optimal_fscore_multi(val['stage2']['y'], y_proba_val_2, train["stage2"]["y_n"].columns, start_step=0.5, stop_step=1.0)
        y_pred_val_2 = np.where(np.max(y_proba_val_2, axis=1) > f_best["f1_weighted_threshold"], train["stage2"]["y_n"].columns[np.argmax(y_proba_val_2, axis=1)], 'Unknown')
        y_pred_test_2 = np.where(np.max(y_proba_test_2, axis=1) > f_best["f1_weighted_threshold"], train["stage2"]["y_n"].columns[np.argmax(y_proba_test_2, axis=1)], 'Unknown')
    else:
        f = open(f'results/stage2/{row["algo"]}/{row["run_id"]}/models/model_{row["number"]}.p', 'rb')
        model_2 = pickle.load(f)
        f.close()
        y_proba_val_2 = model_2.predict_proba(val['stage2']['x_s'])
        y_proba_test_2 = model_2.predict_proba(test['stage2_s'])
        fmacro, fweight, threshold_fscore, f_best = util.optimal_fscore_multi(val['stage2']['y'], y_proba_val_2, model_2.classes_)
        y_pred_val_2 = np.where(np.max(y_proba_val_2, axis=1) > f_best["f1_weighted_threshold"], model_2.classes_[np.argmax(y_proba_val_2, axis=1)], 'Unknown')
        y_pred_test_2 = np.where(np.max(y_proba_test_2, axis=1) > f_best["f1_weighted_threshold"], model_2.classes_[np.argmax(y_proba_test_2, axis=1)], 'Unknown')

    # Collect validation metrics
    val_metric = {
        "algo": row["algo"],
        "run_id": row["run_id"],
        "trial_id": row["number"]
    }
    test_metric = val_metric.copy()

    val_metric["f1_macro"] = f_best["f1_macro"]
    val_metric["f1_weighted"] = f_best["f1_weighted"]
    val_metric['accuracy'] = accuracy_score(val['stage2']['y'], y_pred_val_2)
    val_metric['balanced_accuracy'] = balanced_accuracy_score(val['stage2']['y'], y_pred_val_2)

    test_metric["f1_macro"] = f1_score(test["y_unknown_all"], y_pred_test_2, average='macro')
    test_metric["f1_weighted"] = f1_score(test["y_unknown_all"], y_pred_test_2, average='weighted')
    test_metric['accuracy'] = accuracy_score(test["y_unknown_all"], y_pred_test_2)
    test_metric['balanced_accuracy'] = balanced_accuracy_score(test["y_unknown_all"], y_pred_test_2)


    # Save predictions to dictionary
    predictions['stage2'].append(y_proba_test_2)
    thresholds["stage2"].append(f_best['f1_weighted_threshold'])
    val_results_2.append(val_metric)
    test_results_2.append(test_metric)
    
# Save validation metrics stage 2 on disk
df = pd.DataFrame(val_results_2)
df.to_csv("results/val_stage2.csv", index=False)
# Save test metrics stage 2 on disk
df = pd.DataFrame(test_results_2)
df.to_csv("results/test_stage2.csv", index=False)

# write predictions and thresholds to disk
f = open("results/predictions.pkl","wb")
pickle.dump(predictions, f)
f.close()
f = open("results/thresholds.pkl","wb")
pickle.dump(thresholds, f)
f.close()