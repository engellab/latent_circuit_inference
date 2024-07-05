import json
import os
import numpy as np
import sys
from datetime import date
from pathlib import Path
home = str(Path.home())
if home == '/home/pt1290':
    projects_folder = home
elif home == '/Users/tolmach':
    projects_folder = home + '/Documents/GitHub/'
else:
    pass

LC_configs_path = projects_folder + '/latent_circuit_inference/data/configs'
date = ''.join((list(str(date.today()).split("-"))[::-1]))

activation_name = 'relu'
max_iter = 500
N = 8
num_inputs = 6
num_outputs = 2
tol = 1e-8
lr = 0.02
lambda_w = 0.08
sigma_rec = 0.03
sigma_inp = 0.03
dt = 1
tau = 10
Qinitialization = False
encoding = True
tag = f'{N}units_CDDM{activation_name}'
config_dict = {}
config_dict["N"] = N
config_dict["N_PCs"] = 20
config_dict["num_inputs"] = num_inputs
config_dict["num_outputs"] = num_outputs
W_inp = np.zeros((N, num_inputs))
W_inp[:num_inputs, :num_inputs] = np.eye(num_inputs)
W_out = np.random.rand(num_outputs, N)
config_dict["W_inp"] = W_inp.tolist()
config_dict["W_out"] = W_out.tolist()
config_dict["activation"] = activation_name
config_dict["dt"] = dt
config_dict["tau"] = tau
config_dict["sigma_rec"] = sigma_rec
config_dict["sigma_inp"] = sigma_inp
config_dict["inp_connectivity_mask"] = W_inp.tolist()
config_dict["rec_connectivity_mask"] = np.ones((N, N)).tolist()
config_dict["out_connectivity_mask"] = np.ones((num_outputs, N)).tolist() # no constraint on outputs
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