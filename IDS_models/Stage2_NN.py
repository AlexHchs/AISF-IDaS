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
from sklearn.metrics import auc, roc_curve, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import pathlib
import pickle
import matplotlib.pyplot as plt

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

# Load Data
clean_dir = "/project/data/cicids2017/clean/"
n_benign_val = 1500
x_benign_train, y_benign_train, x_benign_val, y_benign_val, x_benign_test, y_benign_test, x_malicious_train, y_malicious_train, x_malicious_test, y_malicious_test, attack_type_train, attack_type_test, attack_type = util.load_data(clean_dir, sample_size=1948, train_size=n_benign_val, val_size=6815, test_size=56468)

# Input data
algorithm = "nn" # "rf"
neptune_project = "Two-Stage-Model"
dataset = "cic-ids-2017"
stage = "stage2" # "stage2"

x_train, x_val, y_train, y_val = train_test_split(x_malicious_train, y_malicious_train, stratify=attack_type_train, test_size=1500, random_state=42, shuffle=True) # better use stratify=attack_type_train ipv y_malicious_train

y_train_n = pd.get_dummies(y_train)
y_val_n = pd.get_dummies(y_val)

x_val_unk = np.concatenate((x_val, x_benign_train))
y_val_unk = np.concatenate((y_val, np.full(n_benign_val, "Unknown")))
y_val_unk_n = pd.get_dummies(y_val_unk)

scaler = QuantileTransformer(output_distribution='normal')
# scaler = QuantileTransformer(output_distribution='uniform') # uniform instead of normal with range [0,1]
x_train_s = scaler.fit_transform(x_train)
x_val_s = scaler.transform(x_val)
x_val_unk_s = scaler.transform(x_val_unk)

run = neptune.init(project=f'verkerken/{neptune_project}', tags=[dataset, algorithm, stage, 'macro', 'normal'], api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMGJlYTgzNzEtM2U3YS00ODI5LWEzMzgtM2M0MjcyMDIxOWUwIn0=')


run_id = run['sys/id'].fetch()
save_dir = f"results/{stage}/{algorithm}/{run_id}"
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)


def create_nn(params):
    input_layer = Input(shape=(params["n_neurons"][0],))
    model = input_layer
    
    for n in params["n_neurons"][1:-1]:
        model = Dense(n, activation=params['hidden_activation'], activity_regularizer=l2(params["l2_reg"]))(model)
    
    model = Dense(params["n_neurons"][-1], activation=params['output_activation'], activity_regularizer=l2(params['l2_reg']))(model)
    nn = Model(inputs=input_layer, outputs=model)
    nn.compile(optimizer=params['optimizer'], loss=params['loss'], metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return nn

params_nn = {
    'scaler': "quantile", 
    'output_activation': 'linear', # 'softmax',
    "hidden_activation": 'relu',
    "optimizer": "adam",
    "loss": "mean_squared_error", # "categorical_crossentropy",
    "n_neurons": [],
    "l2_reg": 1
}

def objective_nn(trial):
    params = params_nn.copy()
    params['trial_id'] = trial.number
    n_layers = trial.suggest_int("n_layers", 1, 6)
    params['n_neurons'] = [x_train_s.shape[1]]
    for i in range(n_layers):
        params['n_neurons'].append(trial.suggest_int(f'n_layer_{i}', 5, max(5, params['n_neurons'][-1])))
    params['n_neurons'].append(5)
    params["l2_reg"] = trial.suggest_loguniform('l2', 1e-10, 1e-1)
    print(params)
    model = create_nn(params)
    history = model.fit(
        x_train_s,
        y_train_n,
        validation_data=(x_val_s, y_val_n),
        epochs=50,
        shuffle=True,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            min_delta=0.01, 
            mode='min', 
            restore_best_weights=True, 
            verbose=1
        )]
    )
    
    y_pred = model.predict(x_val_unk_s)
    
    # Find optimal threshold for unknown class with F1 score (macro & weighted)
    fmacro, fweight, thresholds, f_best = util.optimal_fscore_multi(y_val_unk, y_pred, y_train_n.columns, start_step=0.5, stop_step=1.0)
    fig = util.plot_f_multi(fmacro, fweight, thresholds, f_best)

    # Plot confusion matrix for optimal threshold
    y_pred_weight = np.where(np.max(y_pred, axis=1) > f_best["f1_weighted_threshold"], y_train_n.columns[np.argmax(y_pred, axis=1)], 'Unknown')
    y_pred_macro = np.where(np.max(y_pred, axis=1) > f_best["f1_macro_threshold"], y_train_n.columns[np.argmax(y_pred, axis=1)], 'Unknown')

    classes = ['(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
    fig_weight = util.plot_confusion_matrix(y_val_unk, y_pred_weight, values=classes, labels=classes)
    fig_macro = util.plot_confusion_matrix(y_val_unk, y_pred_macro, values=classes, labels=classes)
    
    # Add accuracy to metrics
    f_best['accuracy_macro'] = accuracy_score(y_val_unk, y_pred_macro)
    f_best['accuracy_weight'] = accuracy_score(y_val_unk, y_pred_weight)
    f_best['balanced_accuracy_macro'] = balanced_accuracy_score(y_val_unk, y_pred_macro)
    f_best['balanced_accuracy_weight'] = balanced_accuracy_score(y_val_unk, y_pred_weight)
    
    # Log metrics to neptune
    run["metrics"].log(f_best)
    run["params"].log(params)
    run[f"trials/{trial.number}"] = f_best
    run[f"trials/{trial.number}/fscore_plot"].upload(fig)
    run[f"trials/{trial.number}/confusion_weight"].upload(fig_weight)
    run[f"trials/{trial.number}/confusion_macro"].upload(fig_macro)
    run[f"trials/{trial.number}/history"] = history.history
    run["f1_macro"].log(f_best['f1_macro'])
    run["f1_weight"].log(f_best['f1_weighted'])
    
    trial.set_user_attr("f1_macro", f_best['f1_macro'])
    trial.set_user_attr("f1_weight", f_best['f1_weighted'])
    trial.set_user_attr("accuracy_macro", f_best['accuracy_macro'])
    trial.set_user_attr("accuracy_weight", f_best['accuracy_weight'])
    trial.set_user_attr("balanced_accuracy_macro", f_best['balanced_accuracy_macro'])
    trial.set_user_attr("balanced_accuracy_weight", f_best['balanced_accuracy_weight'])
    
    # Save scikit-learn model locally and upload to neptune
    
    # Save Keras model locally and upload to neptune
    model.save(f'{save_dir}/models/model_{trial.number}.h5')
    run[f"trials/{trial.number}/model"].upload(f'{save_dir}/models/model_{trial.number}.h5')
    plt.close('all')
    
    return f_best['f1_macro']

study = optuna.create_study(
    study_name=run_id, 
    direction='maximize', 
    sampler=TPESampler(n_startup_trials=400, n_ei_candidates=24, multivariate=True),
    storage='mysql://optuna:optuna@localhost/optuna_db', 
    load_if_exists=True
)
study.optimize(objective_nn, n_trials=600)

results = study.trials_dataframe()
results.sort_values(by="value", inplace=True, ascending=False)
results.to_csv(f"{save_dir}/results.csv")

run['results_df'].upload(File.as_html(results))
run['optuna/study'].upload(File.as_pickle(study))

run['optuna/param_importances_2'].upload(vis.plot_param_importances(study))
run['optuna/optimization_history'].upload(vis.plot_optimization_history(study))
run['optuna/param_slice'].upload(vis.plot_slice(study))
run['optuna/parallel_coordinate'].upload(vis.plot_parallel_coordinate(study))
run['optuna/param_contour'].upload(vis.plot_contour(study))
