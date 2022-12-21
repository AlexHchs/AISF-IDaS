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
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, roc_curve
import pathlib
import pickle

import neptune.new as neptune
from neptune.new.types import File

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import ThresholdPruner
import optuna.visualization as vis

import matplotlib.pyplot as plt

# Load Data
clean_dir = "/project/data/cicids2017/clean/"
x_benign_train, y_benign_train, x_benign_val, y_benign_val, x_benign_test, y_benign_test, x_malicious_train, y_malicious_train, x_malicious_test, y_malicious_test, attack_type_train, attack_type_test, attack_type = util.load_data(clean_dir, sample_size=1948, train_size=10000, val_size=129485, test_size=56468)

x_binary_val = np.concatenate((x_benign_val, x_malicious_train))
y_binary_val = np.concatenate((y_benign_val, np.full(y_malicious_train.shape[0], -1)))

x_binary_test = np.concatenate((x_benign_test, x_malicious_test))
y_binary_test = np.concatenate((y_benign_test, np.full(y_malicious_test.shape[0], -1)))


# Input data
algorithm = "ocsvm" # "ae"
neptune_project = "Two-Stage-Model"
dataset = "cic-ids-2017"
stage = "stage1" # "stage2"

params_ocsvm = {
    "pca__n_components": 0,
    "ocsvm__kernel": "rbf",
    "ocsvm__gamma": 0,
    "ocsvm__nu": 0
}

scaler = QuantileTransformer(output_distribution='normal')
x_binary_train_s = scaler.fit_transform(x_benign_train)
x_binary_val_s = scaler.transform(x_binary_val)
# x_binary_test_s = scaler.transform(x_binary_test)


# In[11]:


run = neptune.init(project=f'verkerken/{neptune_project}', tags=[dataset, algorithm, stage], api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMGJlYTgzNzEtM2U3YS00ODI5LWEzMzgtM2M0MjcyMDIxOWUwIn0=')

run_id = run['sys/id'].fetch()
save_dir = f"results/{stage}/{algorithm}/{run_id}"
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(save_dir + "/models").mkdir(exist_ok=True)

def create_ocsvm(params):
    return Pipeline(
        [
            ("pca", PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)), 
            ("ocsvm", OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=True, max_iter=-1))
        ]
    ).set_params(**params)

def objective_ocsvm(trial):
    params = params_ocsvm.copy()
    params["pca__n_components"] = trial.suggest_int('n_comp', 1, x_binary_train_s.shape[1])
    params["ocsvm__gamma"] = trial.suggest_float("gamma", 1e-4, 1, log=True)
    params["ocsvm__nu"] = trial.suggest_float("nu", 1e-4, 1, log=True)
    
    model = create_ocsvm(params)
    params["scaler"] = "quantile"
    
    model.fit(x_binary_train_s)
    scores = model.decision_function(x_binary_val_s)
    val_metrics = {}
    if np.isnan(scores).any():
        number_of_nan = np.isnan(scores).sum()
        print(f"Validation scores contains {str(number_of_nan)} nan's")
    else:
        curves_metrics, summary_metrics = util.evaluate_proba(y_binary_val, -scores)
        fig = util.plot_fscores(curves_metrics, summary_metrics)

        # Collect all metrics
        metrics = {
            "trial_id": trial.number,
            "au_precision_recall": auc(curves_metrics['recall'], curves_metrics['precision'])
        }
        for index, row in summary_metrics.iterrows():
            metrics[row['metric']] = row['value']
        fpr, tpr, thresholds = roc_curve(y_binary_val, -scores, pos_label=-1)
        metrics["AUROC"] = auc(fpr, tpr)

        # Save metrics in Optuna
        for k, v in metrics.items():
            trial.set_user_attr(k, v)

        # Log to neptune
        run[f"trials/{trial.number}"] = metrics
        run[f"trials/{trial.number}/params"] = params
        run[f"trials/{trial.number}/fscores_table"].upload(File.as_html(summary_metrics))
        run[f"trials/{trial.number}/fscores_figure"].upload(fig)
        run["AUROC"].log(metrics["AUROC"])
        
        # Save scikit-learn model locally and upload to neptune
        f = open(f'{save_dir}/models/model_{trial.number}.p', 'wb')
        pickle.dump(model, f)
        f.close()
        run[f"trials/{trial.number}/model"].upload(f'{save_dir}/models/model_{trial.number}.p')
        plt.close('all')

    return metrics['AUROC']

study = optuna.create_study(
    study_name=run_id, 
    direction='maximize', 
    sampler=TPESampler(n_startup_trials=500, n_ei_candidates=24, multivariate=True),
    pruner=ThresholdPruner(lower=0.6),
    storage='mysql://optuna:optuna@localhost/optuna_db', 
    load_if_exists=True
)
study.optimize(objective_ocsvm, n_trials= 700)

results = study.trials_dataframe()
results.sort_values(by="value", inplace=True, ascending=False)
results.to_csv(f"{save_dir}/results.csv")

run['results_df'].upload(File.as_html(results))
run['optuna/study'].upload(File.as_pickle(study))

run['optuna/param_importances'].upload(vis.plot_param_importances(study))
run['optuna/optimization_history'].upload(vis.plot_optimization_history(study))
run['optuna/param_slice'].upload(vis.plot_slice(study))
run['optuna/parallel_coordinate'].upload(vis.plot_parallel_coordinate(study))
run['optuna/param_contour'].upload(vis.plot_contour(study))

