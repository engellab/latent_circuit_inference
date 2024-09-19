import sys
sys.path.append("../../../")
# from rnn_coach.latent_circuit_inference.datajoint.datajoint_config import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import torch
import pickle
from rnn_coach.src.RNN_torch import *
from rnn_coach.src.DynamicSystemAnalyzer import *
from rnn_coach.src.RNN_numpy import *
from rnn_coach.src.Task import *
from rnn_coach.src.DataSaver import *
from latent_circuit_inference.src.LatentCircuit import *
from latent_circuit_inference.src.LatentCircuitFitter import *
from latent_circuit_inference.src.LCAnalyzer import *
from latent_circuit_inference.src.utils import *
from latent_circuit_inference.src.circuit_vizualization import *

def mse_scoring(x, y):
    return np.mean((x - y) ** 2)

def R2(x, y):
    return 1.0 - mse_scoring(x, y)/np.var(y)
#
# rnn_dj  = RNNDJ()
# task_dj  = TaskDJ()
# trainer_dj  = TrainerDJ()
# networks_sorted = (rnn_dj & "activation_name = 'relu'").fetch('timestamp', order_by='mse_score')
# networks_sorted = (rnn_dj & "activation_name = 'relu'").fetch('timestamp', order_by='mse_score')
# print(networks_sorted[:50])

# other way
data_path = os.path.join("../../", "../", "rnn_coach", "data", "trained_RNNs", "CDDM")
folders = os.listdir(data_path)
filtered_folders = []
for folder in folders:
    if len(folder.split("_")[0]) != len(folder) and folder != ".DS_Store" and not "exemplar" in folder:
        filtered_folders.append(folder)
tuples = []
for folder in filtered_folders:
    score = float(folder.split("_")[0])
    tuples.append([score, folder])

array = np.array(tuples)
array = pd.DataFrame(array).sort_values(0, ascending=True).to_numpy()
print(array[:20, 1].tolist())
