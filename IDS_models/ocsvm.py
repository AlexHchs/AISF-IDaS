from numpy.random import seed
seed(42)
import numpy as np
import pandas as pd
import util.common as util
import pathlib
import pickle

import neptune
from neptunecontrib.monitoring.optuna import NeptuneCallback, log_study_info
from neptunecontrib.api.table import log_table

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import ThresholdPruner
from optuna.exceptions import StorageInternalError

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline


# Load Data
clean_dir = "/project/data/cicids2017/clean/"
x_binary_train, y_binary_train, x_binary_val, y_binary_val, x_binary_test, y_binary_test, x_multi_train, y_multi_test = util.load_data(clean_dir, train_size=10000, sample_size=1948)

# Set parameters
global_params = {
#     "scaler": None,
    "pca__n_components": 0,
    "ocsvm__kernel": "rbf",
    "ocsvm__gamma": 0,
    "ocsvm__nu": 0
}

# Normalise data
scalers = {
    'standard': StandardScaler(),
    'robust': RobustScaler(quantile_range=(25, 75)),
    'quantile': QuantileTransformer(output_distribution='normal'),
    'minmax': MinMaxScaler(feature_range=(0, 1), copy=True)
}
x_train = {}
x_val = {}
for key, value in scalers.items():
    x_train[key] = value.fit_transform(x_binary_train)
    x_val[key] = value.transform(x_binary_val)

# Link neptune
neptune.init(project_qualified_name='verkerken/two-stage-binary', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMGJlYTgzNzEtM2U3YS00ODI5LWEzMzgtM2M0MjcyMDIxOWUwIn0=')

experiment = neptune.create_experiment('ocsvm-17', tags=["ocsvm", "cicids2017", "binary"])
study_name = experiment.id
save_dir = f'results/binary/ocsvm/{study_name}'
study_storage = 'results/binary/ocsvm.db'
pathlib.Path(f"{save_dir}/models").mkdir(parents=True, exist_ok=True)

def create_model(params):
    return Pipeline(
        [
            ("pca", PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)), 
            ("ocsvm", OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=True, max_iter=-1))
        ]
    ).set_params(**params)

def objective(trial):
    params = global_params.copy()
    params["pca__n_components"] = trial.suggest_int('n_comp', 1, x_binary_train.shape[1])
    params["ocsvm__gamma"] = trial.suggest_float("gamma", 1e-4, 1, log=True)
    params["ocsvm__nu"] = trial.suggest_float("nu", 1e-4, 1, log=True)
    
    model = create_model(params)
    params["scaler"] = trial.suggest_categorical('scaler', scalers.keys())
    
    model.fit(x_train[params["scaler"]])
    scores = model.decision_function(x_val[params["scaler"]])
    val_metrics = {}
    if np.isnan(scores).any():
        number_of_nan = np.isnan(scores).sum()
        print(f"Validation scores contains {str(number_of_nan)} nan's")
        neptune.log_text("warn", f"Validation scores contains {str(number_of_nan)} nan's")
    else:
        val_metrics = util.evaluate_results(y_binary_val, -scores)
        # Save Keras model locally
        f = open(f'{save_dir}/models/model_{trial.number}.p', 'wb')
        pickle.dump(model, f)
        f.close()
        # Save model to neptune server
        neptune.log_artifact(f'{save_dir}/models/model_{trial.number}.p')
        val_metrics.index = val_metrics.index.map(''.join)
        metrics_dict = val_metrics.to_dict()
        neptune.log_text("metrics", str(metrics_dict))
        neptune.log_text("hyperparameters", str(params))
        for metric, value in metrics_dict.items():
            neptune.log_metric(metric, value)
            trial.set_user_attr(metric, value)
    return val_metrics['auroc']
    
study = optuna.create_study(
    study_name=study_name, 
    direction='maximize', 
    sampler=TPESampler(n_startup_trials=200, n_ei_candidates=24, multivariate=True),
    pruner=ThresholdPruner(lower=0.5),
    storage=f'sqlite:///{study_storage}',
    load_if_exists=True
)
study.optimize(objective, timeout=60*60*3, callbacks=[NeptuneCallback()], catch=(StorageInternalError,), n_jobs=-1)

# Save results
log_study_info(study)
results = study.trials_dataframe()
results.sort_values(by=['value'], inplace=True, ascending=False)
results.to_csv(f'{save_dir}/result.csv')
log_table("results_overview", results)