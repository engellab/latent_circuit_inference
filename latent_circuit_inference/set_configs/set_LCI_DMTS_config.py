import json
import os
import numpy as np
import sys
from datetime import date
from pathlib import Path
import pickle
home = str(Path.home())
if home == '/home/pt1290':
    projects_folder = home
elif home == '/Users/tolmach':
    projects_folder = home + '/Documents/GitHub/'
else:
    pass

LC_configs_path = projects_folder + '/latent_circuit_inference/data/configs'
date = ''.join((list(str(date.today()).split("-"))[::-1]))

max_iter = 4000
N = 11
num_inputs = 3
num_outputs = 2
tol = 1e-8
lr = 0.005
lambda_w = 0.001
sigma_rec = 0.03
sigma_inp = 0.03
dt = 1
tau = 10
Qinitialization = False
encoding = True
# tag = f'{N}nodes;{"encoding" if encoding else "decoding"}'
tag = 'DMTS'
config_dict = {}
config_dict["N"] = N
config_dict["N_PCs"] = 32
config_dict["num_inputs"] = num_inputs
config_dict["num_outputs"] = num_outputs


# load file from clustering analysis:
data = pickle.load(open("../../../relu vs tanh RNNs/data/effective_trajectories/traj_DMTS_1708979386.730748.pkl", "rb+"))
traces = data["traces"]
W_inp = np.nan_to_num(data["W_inp"])
W_rec = np.nan_to_num(data["W_rec"])
W_out = np.nan_to_num(data["W_out"])

W_inp = np.squeeze(W_inp * [W_inp >= 0.07])
W_rec = np.squeeze(W_rec * (np.abs(W_rec) >= 0.07))
W_out = np.squeeze(W_out * (np.abs(W_out) >= 0.07))

config_dict["W_inp"] = W_inp.tolist()
config_dict["W_out"] = W_out.tolist()
config_dict["W_rec"] = W_rec.tolist()
config_dict["activation"] = 'relu'
config_dict["dt"] = dt
config_dict["tau"] = tau
config_dict["sigma_rec"] = sigma_rec
config_dict["sigma_inp"] = sigma_inp
Dmask = np.sum(W_rec, axis = 0) >= 0
Dmask = (Dmask.astype(float) * 2 - 1).astype("float32")
config_dict["dale_mask"] = Dmask.tolist()
config_dict["inp_connectivity_mask"] = (W_inp != 0).astype(float).tolist()
config_dict["rec_connectivity_mask"] = (W_rec != 0).astype(float).tolist()
config_dict["out_connectivity_mask"] = (W_out != 0).astype(float).tolist()
config_dict["seed"] = None
config_dict["max_iter"] = max_iter
config_dict["tol"] = tol
config_dict["lr"] = lr
config_dict["lambda_w"] = lambda_w
config_dict["Qinitialization"] = Qinitialization
config_dict["encoding"] = encoding
config_dict["tag"] = tag
json_obj = json.dumps(config_dict, indent=4)
outfile = open(os.path.join(LC_configs_path, f"LCI_config_{tag}.json"), mode="w+")
outfile.write(json_obj)