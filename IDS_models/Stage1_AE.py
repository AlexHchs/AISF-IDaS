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
from sklearn.metrics import auc, roc_curve
import pathlib

# from keras.models import Model
# from keras.layers import Dense, Input
# from keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l2
from util.AUROCEarlyStoppingPruneCallback import AUROCEarlyStoppingPruneCallback

import neptune.new as neptune
from neptune.new.types import File

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import ThresholdPruner
import optuna.visualization as vis

import matplotlib.pyplot as plt

# Load Data
clean_dir = "/project/data/cicids2017/clean/"
x_benign_train, y_benign_train, x_benign_val, y_benign_val, x_benign_test, y_benign_test, x_malicious_train, y_malicious_train, x_malicious_test, y_malicious_test, attack_type_train, attack_type_test, attack_type = util.load_data(clean_dir, sample_size=1948, val_size=129485, test_size=56468)

x_binary_val = np.concatenate((x_benign_val, x_malicious_train))
y_binary_val = np.concatenate((y_benign_val, np.full(y_malicious_train.shape[0], -1)))

x_binary_test = np.concatenate((x_benign_test, x_malicious_test))
y_binary_test = np.concatenate((y_benign_test, np.full(y_malicious_test.shape[0], -1)))

# Input data
algorithm = "ae" # "ocsvm"
neptune_project = "Two-Stage-Model"
dataset = "cic-ids-2017"
stage = "stage1"

params_ae = {
    'scaler': "quantile", 
    'output_activation': 'linear',
    "hidden_activation": 'relu',
    "optimizer": "adam",
    "loss": "mean_squared_error",
    "input_dimension": 0,
    "n_neurons": [],
    "l2_reg": 1
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


def create_ae(params):
    input_layer = Input(shape=(params["input_dimension"],))
    model = input_layer
    
    # Encoder
    for n in params["n_neurons"]:
        model = Dense(n, activation=params['hidden_activation'], activity_regularizer=l2(params["l2_reg"]))(model)

    # Decoder - Do not repeat encoded layer
    for n in reversed(params["n_neurons"][:-1]):
        model = Dense(n, activation=params['hidden_activation'], activity_regularizer=l2(params["l2_reg"]))(model)

    # Output Layer
    model = Dense(params["input_dimension"], activation=params['output_activation'], activity_regularizer=l2(params["l2_reg"]))(model)
    autoencoder = Model(inputs=input_layer, outputs=model)
    autoencoder.compile(optimizer=params['optimizer'], loss=params['loss'])
    return autoencoder

def objective_ae(trial):
    params = params_ae.copy()
    params['trial_id'] = trial.number
    params["input_dimension"] = x_binary_train_s.shape[1]
    params["n_layers"] = trial.suggest_int('encoder_layers', 1, 6)
    n_neurons = [params["input_dimension"]]
    for i in range(params["n_layers"]):
        n_neurons.append(trial.suggest_int(f'n_layer_{i}', 1, max(1, n_neurons[-1] - 1)))
    params["n_neurons"] = n_neurons[1:]
    params["l2_reg"] = trial.suggest_loguniform('l2', 1e-10, 1e-1)

    model = create_ae(params)
    history = model.fit(
        x_binary_train_s,
        x_binary_train_s,
        epochs=15, 
        shuffle=True,
        verbose=0,
        callbacks=[
            AUROCEarlyStoppingPruneCallback(
                x_binary_val_s, 
                y_binary_val, 
                trial,
                min_delta=0.001,
                patience=3,
                mode='max',
                restore_best_weights=True,
                verbose=1
            )
        ]
    )
    x_val_autoencoder = model.predict(x_binary_val_s)
    val_score = util.anomaly_scores(x_binary_val_s, x_val_autoencoder)
    curves_metrics, summary_metrics = util.evaluate_proba(y_binary_val, val_score)
    fig = util.plot_fscores(curves_metrics, summary_metrics)
    
    # Collect all metrics
    metrics = {
        "trial_id": trial.number,
        "au_precision_recall": auc(curves_metrics['recall'], curves_metrics['precision'])
    }
    for index, row in summary_metrics.iterrows():
        metrics[row['metric']] = row['value']
    fpr, tpr, thresholds = roc_curve(y_binary_val, val_score, pos_label=-1)
    metrics["AUROC"] = auc(fpr, tpr)
    metrics["epochs"] = len(history.history['loss'])
    
    # Save metrics in Optuna
    for k, v in metrics.items():
        trial.set_user_attr(k, v)
    
    # Log to neptune
    run[f"trials/{trial.number}"] = metrics
    run[f"trials/{trial.number}/params"] = params
    run[f"trials/{trial.number}/fscores_table"].upload(File.as_html(summary_metrics))
    run[f"trials/{trial.number}/fscores_figure"].upload(fig)
    run[f"trials/{trial.number}/history"] = history.history
    run["AUROC"].log(metrics["AUROC"])
    
    # Save Keras model locally and upload to neptune
    model.save(f'{save_dir}/models/model_{trial.number}.h5')
    run[f"trials/{trial.number}/model"].upload(f'{save_dir}/models/model_{trial.number}.h5')
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
study.optimize(objective_ae, n_trials= 700)

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

