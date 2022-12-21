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
from sklearn.metrics import auc, roc_curve, accuracy_score, balanced_accuracy_score, f1_score, recall_score
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

# import neptune.new as neptune
# from neptune.new.types import File

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

# read predictions and thresholds to disk
f = open("results/predictions.pkl","rb")
predictions = pickle.load(f)
f.close()
f = open("results/thresholds.pkl","rb")
thresholds = pickle.load(f)
f.close()

# Open Neptune
# run = neptune.init(project="verkerken/Two-Stage", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwYmVhODM3MS0zZTdhLTQ4MjktYTMzOC0zYzQyNzIwMjE5ZTAifQ==")

# Combine Stage 1 and Stage 2 for full model performance

# results_test = {
#     "stage1": {},
#     "stage2": {},
#     "extension": {},
#     "full_model": {}
# }

results_12 = []
results_123 = []

classes_2 = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
                                
for index_1, y_proba_1 in enumerate(predictions["stage1"]):
    for (metric, fpr), threshold_b in thresholds["stage1"][index_1].items():
        y_pred_1_n = np.where(y_proba_1 < threshold_b, 1, -1)
        # confusion_1_binary = util.plot_confusion_matrix(test['y_n'], y_pred_1_n, values=[1, -1], labels=["Benign", "Fraud"])
        y_pred_1 = np.where(y_proba_1 < threshold_b, "Benign", "Fraud")
        # results_test['stage1'][(index_1, metric)] = y_pred_1
        y_pred = y_pred_1.astype(object).copy()
        for index_2, y_proba_2 in enumerate(predictions['stage2']):
            print(index_1, index_2)
            threshold_m = thresholds["stage2"][index_2]
            y_pred_2 = np.where(np.max(y_proba_2[y_pred_1 == "Fraud"], axis=1) > threshold_m, train["stage2"]["y_n"].columns[np.argmax(y_proba_2[y_pred_1 == "Fraud"], axis=1)], 'Unknown')
            # confusion_2_multi = util.plot_confusion_matrix(test['y_unknown'][y_pred_1 == "Fraud"], y_pred_2, values=classes_2, labels=classes_2)
            
            # results_test['stage2'][(index_1, metric, index_2)] = y_pred_2
            y_pred[y_pred_1 == "Fraud"] = y_pred_2
            # confusion_12_multi = util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes_2, labels=classes_2)
            result_12 = {
                "index_1": index_1,
                "threshold_b_fpr": fpr,
                "threshold_b_metric": metric,
                "index_2": index_2,
                "bACC": balanced_accuracy_score(test['y_unknown'], y_pred),
                "ACC": accuracy_score(test['y_unknown'], y_pred),
                "f1_micro": f1_score(test['y_unknown'], y_pred, average='micro'),
                "f1_macro": f1_score(test['y_unknown'], y_pred, average='macro'),
                "f1_weighted": f1_score(test['y_unknown'], y_pred, average='weighted')
            }
            results_12.append(result_12)
            # run[f"result12/{index_1}_{metric}_{index_2}"] = result_12
            # run[f"result12/{index_1}_{metric}_{index_2}/confusion_1_binary"].upload(confusion_1_binary)
            # run[f"result12/{index_1}_{metric}_{index_2}/confusion_2_multi"].upload(confusion_2_multi)
            # run[f"result12/{index_1}_{metric}_{index_2}/confusion_12_multi"].upload(confusion_12_multi)
            mask = ((y_pred == "Unknown") & (test['y_unknown_all'] == "Unknown"))
            for quantile, threshold_u in thresholds["extension"][index_1].items():
                y_pred_3 = np.where(y_proba_1[y_pred == "Unknown"] < threshold_u, "Benign", "Unknown")
                y_pred_3_n = np.where(y_proba_1[mask] < threshold_u, 1, -1)
                # confusion_3_multi = util.plot_confusion_matrix(test['y_unknown'][y_pred == "Unknown"], y_pred_3, values=classes_2, labels=classes_2)
                # confusion_3_binary = util.plot_confusion_matrix(test['y_n'][mask], y_pred_3_n, values=[1, -1], labels=["Benign", "Zero-Day"])
                # results_test['extension'][(index_1, metric, index_2, quantile)] = y_pred_3
                
                y_pred_final = y_pred.copy()
                y_pred_final[y_pred == "Unknown"] = y_pred_3
                # confusion_123_multi = util.plot_confusion_matrix(test['y_unknown'], y_pred_final, values=classes_2, labels=classes_2)
                result_123 = {
                    "index_1": index_1,
                    "threshold_b_fpr": fpr,
                    "threshold_b_metric": metric,
                    "index_2": index_2,
                    "quantile": quantile,
                    "bACC": balanced_accuracy_score(test['y_unknown'], y_pred_final),
                    "ACC": accuracy_score(test['y_unknown'], y_pred_final),
                    "f1_micro": f1_score(test['y_unknown'], y_pred_final, average='micro'),
                    "f1_macro": f1_score(test['y_unknown'], y_pred_final, average='macro'),
                    "f1_weighted": f1_score(test['y_unknown'], y_pred_final, average='weighted'),
                    "zero_day_recall_extension": recall_score(test['y_n'][mask], y_pred_3_n, pos_label=-1),
                    "zero_day_recall_total": (y_pred_3_n == -1).sum() / 47
                }
                results_123.append(result_123)
                # results_test['full_model'][(index_1, metric, index_2, quantile)] = y_pred_final
                # run[f"result123/{index_1}_{metric}_{index_2}_{quantile}"] = result_123
                # run[f"result123/{index_1}_{metric}_{index_2}_{quantile}/confusion_1_binary"].upload(confusion_1_binary)
                # run[f"result123/{index_1}_{metric}_{index_2}_{quantile}/confusion_2_multi"].upload(confusion_2_multi)
                # run[f"result123/{index_1}_{metric}_{index_2}_{quantile}/confusion_12_multi"].upload(confusion_12_multi)
                # run[f"result123/{index_1}_{metric}_{index_2}_{quantile}/confusion_3_multi"].upload(confusion_3_multi)
                # run[f"result123/{index_1}_{metric}_{index_2}_{quantile}/confusion_3_binary"].upload(confusion_3_binary)
                # run[f"result123/{index_1}_{metric}_{index_2}_{quantile}/confusion_123_multi"].upload(confusion_123_multi)
    
        # plt.close('all')

    
# Save metrics on disk
df_12 = pd.DataFrame(results_12)
df_12.to_csv("results/results_12.csv", index=False)
df_123 = pd.DataFrame(results_123)
df_123.to_csv("results/results_123.csv", index=False)

# Upload metrics to neptune
# run["results_12"].upload(File.as_html(df_12))
# run["results_123"].upload(File.as_html(df_123))