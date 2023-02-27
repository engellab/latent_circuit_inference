import sys
sys.path.append("../../")
import os
from rnn_coach.src.DynamicSystemAnalyzer import *
from rnn_coach.src.RNN_numpy import *
from rnn_coach.src.Task import *
from rnn_coach.src.PerformanceAnalyzer import *
import numpy as np
import pickle
import json
data_path = os.path.join('../' '../', 'rnn_coach', "data", "trained_RNNs", "CDDM")
folders = os.listdir(data_path)
for i, folder in enumerate(folders):
    if folder == "exemplar_RNN":
        pass
    else:
        if os.path.isdir(os.path.join(data_path, folder)):
            files = os.listdir(os.path.join(data_path, folder))
            if ".DS_Store" in files:
                files.pop(files.index(".DS_Store"))
            score = files[0].split("_")[0]
            if score == 'params':
                score = files[1].split("_")[0]

            # get RNN parameters
            RNN_folder_full_path = os.path.join(data_path, folder)
            rnn_config = json.load(open(os.path.join(RNN_folder_full_path, f"{score}_config.json"), "rb+"))
            try:
                rnn_data = pickle.load(open(os.path.join(RNN_folder_full_path, f"{score}_params_CDDM.pkl"), "rb+"))
            except:
                rnn_data = pickle.load(open(os.path.join(RNN_folder_full_path, f"params_CDDM_{score}.pkl"), "rb+"))
                os.rename(os.path.join(data_path, folder, f"params_CDDM_{score}.pkl"),
                          os.path.join(data_path, folder, f"{score}_params_CDDM.pkl"))
            activation_name = rnn_config["activation"]
            if activation_name == 'relu':
                activation = lambda x: np.maximum(x, 0)
            elif activation_name == 'tanh':
                activation = np.tanh
            elif activation_name == 'sigmoid':
                activation = lambda x: 1 / (1 + np.exp(-x))
            elif activation_name == 'softplus':
                activation = lambda x: np.log(1 + np.exp(5 * x))

            # defining RNN
            N = rnn_data["N"]
            W_rec = rnn_data["W_rec"]
            W_inp = rnn_data["W_inp"]
            W_out = rnn_data["W_out"]
            dt = rnn_data["dt"]
            tau = rnn_data["tau"]
            RNN = RNN_numpy(N=N, W_rec=W_rec, W_inp=W_inp, W_out=W_out, dt=dt, tau=tau,
                            activation=activation)
            RNN.y = np.zeros(N)

            task_data = rnn_config["task_params"]
            n_steps = task_data["n_steps"]
            n_inputs = task_data["n_inputs"]
            n_outputs = task_data["n_outputs"]
            mask = rnn_config["mask"]
            task = TaskCDDM(n_steps=n_steps, n_inputs=n_inputs, n_outputs=n_outputs, task_params=task_data)

            analyzer = PerformanceAnalyzerCDDM(RNN)
            score_function = lambda x, y: np.mean((x - y) ** 2)
            input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
            new_score = analyzer.get_validation_score(score_function, input_batch_valid, target_batch_valid,
                                                      mask, sigma_rec=0, sigma_inp=0)
            new_score = np.round(new_score, 7)
            files = os.listdir(os.path.join(data_path, folder))
            for file in files:
                if score in file:
                    ending = file.split(str(score))[1]
                    os.rename(os.path.join(data_path, folder, f'{score}{ending}'), os.path.join(data_path, folder, f'{new_score}{ending}'))
            print(i, folder)
