from numpy.random import seed
seed(42)
import tensorflow
tensorflow.random.set_seed(42)
import numpy as np
import pandas as pd
import util.common as util
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import pathlib

from keras.models import Model
from keras.layers import Dense, Input
from keras.regularizers import l2
from util.AUROCEarlyStoppingPruneCallback import AUROCEarlyStoppingPruneCallback
from util.DenseTied import DenseTied

import neptune
from neptunecontrib.monitoring.optuna import NeptuneCallback, log_study_info
from neptunecontrib.api.table import log_table

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import ThresholdPruner
from optuna.exceptions import StorageInternalError

# Load Data
clean_dir = "/project/data/cicids2017/clean/"
x_binary_train, y_binary_train, x_binary_val, y_binary_val, x_binary_test, y_binary_test, x_multi_train, y_multi_test = util.load_data(clean_dir, sample_size=1948)

# Set parameters
global_params = {
    "scaler": "quantile", # "minmax"
    "output_activation": "linear", # "sigmoid",
    "hidden_activation": "relu",
    "optimizer": "adam",
    "loss": "mean_squared_error",
    "input_dimension": 0,
    "n_layers": 0,
    "n_neurons": [0],
    "l2_reg": 0
}

# Normalise data
scalers = {
    "minmax": MinMaxScaler(feature_range=(0, 1), copy=True),
    "quantile": QuantileTransformer(output_distribution='normal')
}
binary_scaler = scalers[global_params['scaler']]
x_binary_train_s = binary_scaler.fit_transform(x_binary_train)
x_binary_val_s = binary_scaler.transform(x_binary_val)
# x_binary_test_s = binary_scaler.transform(x_binary_test)

# Link neptune
neptune.init(project_qualified_name='verkerken/two-stage-binary', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMGJlYTgzNzEtM2U3YS00ODI5LWEzMzgtM2M0MjcyMDIxOWUwIn0=')

experiment = neptune.create_experiment('ae-17-linear-cpu-tied', tags=["autoencoder", "cicids2017", "binary", "linear", "CPU", "Tied"])
study_name = experiment.id
save_dir = f'results/binary/autoencoder/{study_name}'
study_storage = 'results/binary/autoencoder_linear_cpu_tied.db'
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)


def create_model(params):
    input_layer = Input(shape=(params["input_dimension"],))
    model = input_layer
    
    # Encoder
    encoder = []
    for n in params["n_neurons"]:
        encoder_layer = Dense(n, activation=params['hidden_activation'], activity_regularizer=l2(params["l2_reg"]))
        encoder.append(encoder_layer)
        model = encoder_layer(model)

    # Decoder - Do not repeat encoded layer
    for n, encoder_layer in zip(reversed(params["n_neurons"][:-1]), reversed(encoder[1:])):
        decoder_layer = DenseTied(n, activation=params['hidden_activation'], tied_to=encoder_layer, activity_regularizer=l2(params["l2_reg"]))
        model = decoder_layer(model)

    # Output Layer
    model = DenseTied(params["input_dimension"], activation=params['output_activation'], tied_to=encoder[0], activity_regularizer=l2(params["l2_reg"]))(model)
    autoencoder = Model(inputs=input_layer, outputs=model)
    autoencoder.compile(optimizer=params['optimizer'], loss=params['loss'])
    return autoencoder

def objective(trial):
    params = global_params.copy()
    params["input_dimension"] = x_binary_train_s.shape[1]
    params["n_layers"] = trial.suggest_int('encoder_layers', 1, 5)
    n_neurons = [params["input_dimension"]]
    for i in range(params["n_layers"]):
        n_neurons.append(trial.suggest_int(f'n_layer_{i}', 1, max(1, n_neurons[-1] - 1)))
    params["n_neurons"] = n_neurons[1:]
    params["l2_reg"] = trial.suggest_loguniform('l2', 1e-10, 1e-1)
    
    autoencoder = create_model(params)
    history = autoencoder.fit(
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
    x_val_autoencoder = autoencoder.predict(x_binary_val_s)
    val_score = util.anomaly_scores(x_binary_val_s, x_val_autoencoder)
    val_metrics = util.evaluate_results(y_binary_val, val_score)
    trial.set_user_attr('epochs', len(history.history['loss']))
    neptune.log_metric('epochs', len(history.history['loss']))
    trial.set_user_attr('losses', history.history['loss'])
    # Save Keras model locally
#     autoencoder.save(f'{save_dir}/models/model_{trial.number}.h5')
    # Save model to neptune server
#     neptune.log_artifact(f'{save_dir}/models/model_{trial.number}.h5')
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
    sampler=TPESampler(n_startup_trials=300, n_ei_candidates=24, multivariate=True),
    pruner=ThresholdPruner(lower=0.5),
    storage=f'sqlite:///{study_storage}',
    load_if_exists=True
)
study.optimize(objective, timeout=60*60*6, callbacks=[NeptuneCallback()], catch=(StorageInternalError,), n_jobs=-1)

# Save results
log_study_info(study)
results = study.trials_dataframe()
results.sort_values(by=['value'], inplace=True, ascending=False)
results.to_csv(f'{save_dir}/result.csv')
log_table("results_overview", results)