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

import numpy as np
import pandas as pd
import util.common as util
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import pathlib

import neptune.new as neptune
from neptune.new.types import File

import optuna
from optuna.samplers import TPESampler, GridSampler
from optuna.pruners import ThresholdPruner
import optuna.visualization as vis

import matplotlib.pyplot as plt

# Input data
algorithm = "rf"
neptune_project = "Two-Stage-Model"
dataset = "cic-ids-2017"
stage = "baseline"

# # Load Data
clean_dir = "/project/data/cicids2017/clean/"
x_benign_train, y_benign_train, x_benign_val, y_benign_val, x_benign_test, y_benign_test, x_malicious_train, y_malicious_train, x_malicious_test, y_malicious_test, attack_type_train, attack_type_test, attack_type = util.load_data(clean_dir, sample_size=1948, train_size=2, val_size=1363, test_size=56468)

x_train = np.concatenate((x_benign_val, x_malicious_train))
y_train = np.concatenate((np.full(len(x_benign_val), "Benign"), y_malicious_train))
x_test = np.concatenate((x_benign_test, x_malicious_test))
y_test = np.concatenate((np.full(len(x_benign_test), "Benign"), y_malicious_test))

y_test[y_test == "Heartbleed"] = "Unknown"
y_test[y_test == "Infiltration"] = "Unknown"


# ## Normalise data

scalers = {
    'standard': StandardScaler(),
    'robust': RobustScaler(quantile_range=(25, 75)),
    'quantile': QuantileTransformer(output_distribution='normal'),
    'minmax': MinMaxScaler(feature_range=(0, 1), copy=True)
}
x_train_s = {}
x_test_s = {}
for key, value in scalers.items():
    x_train_s[key] = value.fit_transform(x_train)
    x_test_s[key] = value.transform(x_test)


# Link Neptune
run = neptune.init(project=f'verkerken/{neptune_project}', tags=[dataset, algorithm, stage], api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMGJlYTgzNzEtM2U3YS00ODI5LWEzMzgtM2M0MjcyMDIxOWUwIn0=')

run_id = run['sys/id'].fetch()
save_dir = f"results/{stage}/{algorithm}/{run_id}"
pathlib.Path(save_dir + "/models").mkdir(parents=True, exist_ok=True)

# search_params = {
# #     'bootstrap': [True, False],
#     'max_features': list(map(float, np.arange(0.01, 1, 0.05))),
#     'max_samples': list(map(float, np.arange(0.01, 1, 0.05))),
#     'min_samples_leaf': list(map(int, np.arange(1, 100, 5))),
# #     'scaler': scalers.keys()
# }

# # Train Random Forest
def create_model(params):
    return RandomForestClassifier(random_state=42, oob_score=True).set_params(**params)

def objective(trial):
    params = {}
#     params["bootstrap"] = trial.suggest_categorical('bootstrap', [True, False])
    params["n_estimators"] = trial.suggest_int("n_estimators", 10, 100)
    params["max_samples"] = trial.suggest_float("max_samples", 0.01, 1)
    params["max_features"] = trial.suggest_float("max_features", 0.01, 1)
    params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 100)
    
    model = create_model(params)
    params["scaler"] = 'quantile' # trial.suggest_categorical('scaler', scalers.keys())
    
    
#     experiment = neptune.create_experiment(f'rf-17', tags=["RandomForest", "cicids2017", "baseline", "CPU"], params=params)
    
    model.fit(x_train_s[params["scaler"]], y_train)
    y_pred = model.predict(x_test_s[params["scaler"]])
    
    # plot confusion matrix
    classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
    confusion_fig = util.plot_confusion_matrix(y_test, y_pred, values=classes, labels=classes)
    
    # compute metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
    metrics['accuracy_excl'] = accuracy_score(y_test[y_test != 'Unknown'], y_pred[y_test != 'Unknown'])
    metrics['balanced_accuracy_excl'] = balanced_accuracy_score(y_test[y_test != 'Unknown'], y_pred[y_test != 'Unknown'])
    metrics['f1_score_macro'] = f1_score(y_test, y_pred, average='macro')
    metrics['f1_score_weighted'] = f1_score(y_test, y_pred, average='weighted')
    # without zero-days 
    metrics['f1_score_macro_excl'] = f1_score(y_test, y_pred, average='macro', labels=model.classes_)
    metrics['f1_score_weighted_excl'] = f1_score(y_test, y_pred, average='weighted', labels=model.classes_)
    metrics['OOB'] = model.oob_score_
    
    # Log metrics to neptune
    run[f'trials/{trial.number}'] = metrics
    run[f'trials/{trial.number}/params'] = params
    run[f"trials/{trial.number}/confusion_matrix"].upload(confusion_fig)
#     run['balanced_accuracy'].log(metrics['balanced_accuracy'])
#     run['accuracy'].log(metrics['accuracy'])
#     run['f1_score_macro'].log(metrics['f1_score_macro'])
#     run['f1_score_weighted'].log(metrics['f1_score_weighted'])
#     run['f1_score_macro_excl'].log(metrics['f1_score_macro_exl'])
#     run['f1_score_weighted_excl'].log(metrics['f1_score_weighted_exl'])
    
    for k, v in metrics.items():
        trial.set_user_attr(k, v)
#     trial.set_user_attr("balanced_accuracy", metrics['balanced_accuracy'])
#     trial.set_user_attr("accuracy", metrics['accuracy'])
#     trial.set_user_attr("balanced_accuracy", metrics['balanced_accuracy'])
#     trial.set_user_attr("accuracy", metrics['accuracy'])
#     trial.set_user_attr("f1_score_macro", metrics['f1_score_macro'])
#     trial.set_user_attr("f1_score_weighted", metrics['f1_score_weighted'])
#     trial.set_user_attr("f1_score_macro_exl", metrics['f1_score_macro_exl'])
#     trial.set_user_attr("f1_score_weighted_exl", metrics['f1_score_weighted_exl'])
    trial.set_user_attr("OOB", model.oob_score_)
    
    # Save model locally and upload to neptune
    f = open(f'{save_dir}/models/model_{trial.number}.p', 'wb')
    pickle.dump(model, f)
    f.close()
    run[f"trials/{trial.number}/model"].upload(f'{save_dir}/models/model_{trial.number}.p')
    plt.close('all')
    
    return model.oob_score_

study = optuna.create_study(
    study_name=run_id, 
    direction='maximize', 
    sampler=TPESampler(n_startup_trials=1000, n_ei_candidates=24, multivariate=True), # GridSampler(search_params),
    storage='mysql://optuna:optuna@localhost/optuna_db', 
    load_if_exists=True
)
study.optimize(objective, n_trials=1200)

# Save results
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