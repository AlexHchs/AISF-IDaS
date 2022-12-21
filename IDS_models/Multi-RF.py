from numpy.random import seed
seed(42)
import tensorflow
tensorflow.random.set_seed(42)

import numpy as np
import pandas as pd
import util.common as util
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.metrics import f1_score
import pathlib

import neptune
from neptunecontrib.monitoring.optuna import NeptuneCallback, log_study_info
from neptunecontrib.api.table import log_table
from neptunecontrib.api import log_chart


import optuna
from optuna.samplers import TPESampler, GridSampler
from optuna.pruners import ThresholdPruner
from optuna.exceptions import StorageInternalError

import matplotlib.pyplot as plt
from tensorflow import keras

# Link Neptune
neptune.init(project_qualified_name='verkerken/two-stage-multi', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMGJlYTgzNzEtM2U3YS00ODI5LWEzMzgtM2M0MjcyMDIxOWUwIn0=')
experiment = neptune.create_experiment('rf-17-multi', tags=["RandomForest", "cicids2017", "multi", "CPU"])
study_name = experiment.id
save_dir = f'results/multi/randomforest/{study_name}'
study_storage = 'results/multi/randomforest-multi.db'
pathlib.Path(f"{save_dir}/models").mkdir(parents=True, exist_ok=True)


clean_dir = "/project/data/cicids2017/clean/"
x_binary_train, y_binary_train, x_binary_val, y_binary_val, x_binary_test, y_binary_test, x_multi_train, y_multi_train, x_multi_test, y_multi_test = util.load_data(clean_dir, train_size=100000, sample_size=1948)

x_train = x_multi_train
y_train = y_multi_train
# x_test = np.concatenate((x_binary_train[-585:], x_multi_test))
# y_test = np.concatenate((np.full(585, "Benign"), y_multi_test))
x_test = x_multi_test
y_test = y_multi_test
y_test[y_test == "Heartbleed"] = "Unknown"
y_test[y_test == "Infiltration"] = "Unknown"


# # Rescale Data


from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler

binary_scaler = QuantileTransformer(output_distribution='normal')
x_binary_train_s = binary_scaler.fit_transform(x_binary_train)
x_binary_val_s = binary_scaler.transform(x_binary_val)
x_train_1 = binary_scaler.transform(x_train)
x_test_1 = binary_scaler.transform(x_test)


# # Load Binary Model
study_name = "TWOS-37"
trial_number = 1223
path = f'results/binary/autoencoder/{study_name}/models/model_{trial_number}.h5'
model = keras.models.load_model(path)
# print(model.summary())

# Get Score feature

x_predict = model.predict(x_train_1)
score_train = util.anomaly_scores(x_train_1, x_predict)

x_predict = model.predict(x_test_1)
score_test = util.anomaly_scores(x_test_1, x_predict)

# ## Concat Anomaly Score as Feature
x_train_2 = np.column_stack((x_train, score_train))
x_test_2 = np.column_stack((x_test, score_test))

# Rescale MultiClass
scalers = {
    'standard': StandardScaler(),
    'robust': RobustScaler(quantile_range=(25, 75)),
    'quantile': QuantileTransformer(output_distribution='normal'),
    'minmax': MinMaxScaler(feature_range=(0, 1), copy=True)
}
x_train_s = {}
x_test_s = {}
for key, value in scalers.items():
    x_train_s[key] = value.fit_transform(x_train_2)
    x_test_s[key] = value.transform(x_test_2)
    

def create_model(params):
    return RandomForestClassifier(random_state=42, oob_score=True).set_params(**params)
    
def objective(trial):
    params = {}
    params["max_samples"] = trial.suggest_float("max_samples", 0.01, 1)
    params["max_features"] = trial.suggest_float("max_features", 0.01, 1)
    params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 100)
    
    model = create_model(params)
    params["scaler"] = trial.suggest_categorical('scaler', scalers.keys())
    
    model.fit(x_train_s[params["scaler"]], y_train)
    y_proba = model.predict_proba(x_test_s[params["scaler"]])
    
    # Save Keras model locally
    f = open(f'{save_dir}/models/model_{trial.number}.p', 'wb')
    pickle.dump(model, f)
    f.close()
    # Save model to neptune server
    neptune.log_artifact(f'{save_dir}/models/model_{trial.number}.p')
    
    thresholds = np.arange(0.0, 1.0, 0.005)
    fscore = np.zeros(shape=(len(thresholds)))
    for index, threshold in enumerate(thresholds):
        # Corrected probabilities
        y_pred = np.where(np.max(y_proba, axis=1) > threshold, model.classes_[np.argmax(y_proba, axis=1)], 'Unknown')
        # Calculate the f-score
        fscore[index] = f1_score(y_test, y_pred, average='macro')
        
    index = np.argmax(fscore)
    thresholdOpt = thresholds[index]
    fscoreOpt = fscore[index]
    neptune.log_metric("OOB", model.oob_score_)
    neptune.log_metric("thresholdOptimal", thresholdOpt)
    neptune.log_metric("f1scoreOptimal", fscoreOpt)
    neptune.log_text("hyperparameters", str(params))
    trial.set_user_attr("thresholdOptimal", thresholdOpt)
    trial.set_user_attr("f1scoreOptimal", fscoreOpt)
    trial.set_user_attr("OOB", model.oob_score_)
    
    
#     threshold = thresholdOpt
#     y_pred = np.where(np.max(y_proba, axis=1) > threshold, model.classes_[np.argmax(y_proba, axis=1)], 'Unknown')
#     classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
#     fig = util.plot_confusion_matrix(y_test, y_pred, values=classes, labels=classes)
#     neptune.log_image("confusion_matrix", fig)
    
#     fig = plt.figure()
#     plt.plot(thresholds, fscore, label="F1 Score")
#     plt.plot(thresholdOpt, fscoreOpt, marker="X")
#     plt.annotate(f"F1: {str(round(fscoreOpt, 3))}", (thresholdOpt, fscoreOpt))
#     plt.xlabel(r"threshold ($\tau_M$)")
#     plt.ylabel("score")
#     plt.legend()
#     neptune.log_image("threshold_score", fig)
    
    return model.oob_score_


study = optuna.create_study(
    study_name=study_name, 
    direction='maximize', 
    sampler=TPESampler(n_startup_trials=100, n_ei_candidates=24, multivariate=True),
    storage=f'sqlite:///{study_storage}',
    load_if_exists=True
)
study.optimize(objective, n_trials=200, callbacks=[NeptuneCallback()], catch=(StorageInternalError,), n_jobs=-1)

# Save results
results = study.trials_dataframe()
results.sort_values(by=['value'], inplace=True, ascending=False)
results.to_csv(f'{save_dir}/result.csv')
log_study_info(study)
log_table("results_overview", results)