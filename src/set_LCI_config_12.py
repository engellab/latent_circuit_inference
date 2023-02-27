import json
import os
import numpy as np
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from latent_circuit_inference.src.utils import get_project_root
from datetime import date
date = ''.join((list(str(date.today()).split("-"))[::-1]))

max_iter = 1250
n = 12
tag = '12-nodes'
num_inputs = 6
num_outputs = 2
tol = 1e-8
lr = 0.05
lambda_w = 0.08
sigma_rec = 0.03
sigma_inp = 0.03
data_folder = os.path.abspath(os.path.join(get_project_root(), "data", "inferred_LCs"))

config_dict = {}
config_dict["n"] = n
config_dict["N"] = 12
config_dict["num_inputs"] = num_inputs
config_dict["num_outputs"] = num_outputs
W_inp = np.zeros((n, num_inputs))
W_inp[:6, :6] = np.eye(6)
W_inp[[6, 7, 8, 9], [2, 3, 4, 5]] = 0.5
W_out = np.zeros((num_outputs, n))
W_out[0, 10] = 1
W_out[1, 11] = 1

config_dict["W_inp"] = W_inp.tolist()
config_dict["W_out"] = W_out.tolist()
config_dict["activation"] = 'relu'
config_dict["dt"] = 1
config_dict["tau"] = 10
config_dict["sigma_rec"] = sigma_rec
config_dict["sigma_inp"] = sigma_inp
config_dict["inp_connectivity_mask"] = (np.abs(W_inp) > 0).tolist()
rec_connectivity_mask = np.ones((n, n))
# no context modulation of auxiliary nodes!
rec_connectivity_mask[6, [0, 1]] = 0
rec_connectivity_mask[7, [0, 1]] = 0
rec_connectivity_mask[8, [0, 1]] = 0
rec_connectivity_mask[9, [0, 1]] = 0
config_dict["rec_connectivity_mask"] = rec_connectivity_mask.tolist()
config_dict["out_connectivity_mask"] = (np.abs(W_out) > 0).tolist()
config_dict["seed"] = None
config_dict["max_iter"] = max_iter
config_dict["tol"] = tol
config_dict["lr"] = lr
config_dict["lambda_w"] = lambda_w
config_dict["data_folder"] = data_folder
config_dict["tag"] = tag
json_obj = json.dumps(config_dict, indent=4)
outfile = open(os.path.join(get_project_root(), "data", "configs", f"LCI_config_{tag}.json"), mode="w")
outfile.write(json_obj)