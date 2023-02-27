import sys
sys.path.append("../../")
# from rnn_coach.src.datajoint.datajoint_config import *
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
from latent_circuit_inference.src.LCPerformanceAnalyzer import *
from latent_circuit_inference.src.utils import *
from latent_circuit_inference.src.circuit_vizualization import *

def mse_scoring(x, y):
    return np.mean((x - y) ** 2)

def R2(x, y):
    return 1.0 - mse_scoring(x, y)/np.var(y)

# rnn_dj  = RNNDJ()
# task_dj  = TaskDJ()
# trainer_dj  = TrainerDJ()
# networks_sorted = (rnn_dj & "activation_name = 'relu'").fetch("rnn_hash", order_by='mse_score')
# top_RNN = networks_sorted[0]
# rnn_data = pd.DataFrame((rnn_dj & f"rnn_hash = '{top_RNN}'").fetch())
# task_string = rnn_data["task_hash"][0]
# task_data = pd.DataFrame((task_dj & f"task_hash='{task_string}'").fetch())
# trainer_string = rnn_data["trainer_hash"][0]
# trainer_data = pd.DataFrame((trainer_dj & f"trainer_hash='{trainer_string}'").fetch())
# rnn_data = pickle.load(open(os.path.join("../", "../", "rnn_coach",
#                                          "data", "trained_RNNs", "CDDM",
#                                          "20230207-08111997", "params_CDDM_0.008362.pkl"), "rb+"))

rnn_data = json.load(open(os.path.join("../", "../", "rnn_coach",
                                         "data", "trained_RNNs", "CDDM",
                                         "20230207-08111997", "0.008362_config.json"), "rb+"))
# taskname = rnn_data["task_name"]
LCI_config_file = json.load(open(os.path.join("../", "data", "configs", f"LCI_config.json"), mode="r", encoding='utf-8'))

# defining RNN:
activation_name = rnn_data["activation_name"][0]
RNN_N = rnn_data["n"][0]
if activation_name == 'relu':
    activation_RNN = lambda x: torch.maximum(x, torch.tensor(0))
elif activation_name == 'tanh':
    activation_RNN = torch.tanh
elif activation_name == 'sigmoid':
    activation_RNN = lambda x: 1/(1 + torch.exp(-x))
elif activation_name == 'softplus':
    activation_RNN = lambda x: torch.log(1 + torch.exp(5 * x))
dt = rnn_data["dt"][0]
tau = rnn_data["tau"][0]
connectivity_density_rec = rnn_data["connectivity_density_rec"][0]
spectral_rad = rnn_data["sr"][0]
sigma_inp = rnn_data["sigma_inp"][0]
sigma_rec = rnn_data["sigma_rec"][0]

seed = LCI_config_file["seed"]
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# rng = torch.Generator(device=torch.device('cpu'))
if not seed is None:
    rng.manual_seed(seed)
input_size = np.array(rnn_data["w_inp"][0]).shape[1]
output_size = np.array(rnn_data["w_out"][0]).shape[0]

# Task:
n_steps = task_data["n_steps"][0]
task_params = task_data["task_params"][0]
task_params["coherences"] = task_data["task_params"][0]["coherences"]

# LC
n = LCI_config_file["n"]
LC_N = LCI_config_file["N"]
W_inp = np.array(LCI_config_file["W_inp"])
W_out = np.array(LCI_config_file["W_out"])
# Fitter:
lambda_w = LCI_config_file["lambda_w"]
max_iter = LCI_config_file["max_iter"]
tol = LCI_config_file["tol"]
lr = LCI_config_file["lr"]
actvation_name = LCI_config_file["activation"]
inp_connectivity_mask = np.array(LCI_config_file["inp_connectivity_mask"])
rec_connectivity_mask = np.array(LCI_config_file["rec_connectivity_mask"])
out_connectivity_mask = np.array(LCI_config_file["out_connectivity_mask"])
if activation_name == 'relu':
    activation_LC = lambda x: torch.maximum(x, torch.tensor(0))
elif activation_name == 'tanh':
    activation_LC = torch.tanh
elif activation_name == 'sigmoid':
    activation_LC = lambda x: 1/(1 + torch.exp(-x))
elif activation_name == 'softplus':
    activation_LC = lambda x: torch.log(1 + torch.exp(5 * x))

# # creating instances:
rnn_torch = RNN_torch(N=RNN_N, dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                      activation=activation_RNN, random_generator=rng, device=device)
RNN_params = {"W_inp":np.array(rnn_data["w_inp"][0]),
              "W_rec":np.array(rnn_data["w_rec"][0]),
              "W_out":np.array(rnn_data["w_out"][0]),
              "b_rec":np.array(rnn_data["b_rec"][0]),
              "y_init":np.zeros(RNN_N)}
rnn_torch.set_params(RNN_params)

lc = LatentCircuit(n=n,
                   N=LC_N,
                   W_inp=torch.Tensor(W_inp),
                   W_out=torch.Tensor(W_out),
                   inp_connectivity_mask=inp_connectivity_mask,
                   rec_connectivity_mask=rec_connectivity_mask,
                   out_connectivity_mask=out_connectivity_mask,
                   activation=activation_LC,
                   device=device,
                   random_generator=rng)

task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lc.parameters(),
                             lr=lr)
fitter = LatentCircuitFitter(LatentCircuit=lc, RNN=rnn_torch, Task=task,
                             max_iter=max_iter, tol=tol,
                             optimizer=optimizer, criterion=criterion,
                             lambda_w=lambda_w)

lc_inferred, train_losses, val_losses, net_params = fitter.run_training()

# defining circuit
n = 8
U = net_params["U"]
q = net_params["q"]
Q = q @ U # should be 100 x 8
W_rec = RNN_params["W_rec"]
w_rec_bar = Q @ W_rec @ Q.T
w_rec = net_params["W_rec"]
names = ["ctx m", "ctx c", "mr", "ml", "cr", "cl", "OutR", "OutL"]
w_rec = net_params["W_rec"]
w_inp = net_params["W_inp"]
w_out = net_params["W_out"]
dt = net_params["dt"]
tau = net_params["tau"]
activation_fun_circuit = lambda x: np.maximum(0, x)
circuit = RNN_numpy(N=n, W_rec=w_rec, W_inp=w_inp, W_out=w_out, dt=dt, tau=tau, activation=activation_fun_circuit)
circuit.y = np.zeros(n)

# defining RNN
N = 100
x = np.random.randn(n)
W_rec = RNN_params["W_rec"]
W_inp = RNN_params["W_inp"]
W_out = RNN_params["W_out"]
dt = net_params["dt"]
tau = net_params["tau"]
activation_fun_RNN = lambda x: np.maximum(0, x)
RNN = RNN_numpy(N=N, W_rec=W_rec, W_inp=W_inp, W_out=W_out, dt=dt, tau=tau, activation=activation_fun_RNN)
RNN.y = np.zeros(n)

# defining analyzer
node_labels = ['ctx m', "ctx c", "mR", "mL", "cR", "cL", "OutR", "OutL"]
analyzer = LCPerformanceAnalyzer(circuit, labels=node_labels)
input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
mask = np.array(task_data["mask"][0])

#MSE mse_score_RNN
score_function = lambda x, y: np.mean((x - y) ** 2)
mse_score = analyzer.get_validation_score(scoring_function=mse_scoring,
                                          input_batch=input_batch_valid,
                                          target_batch=target_batch_valid,
                                          mask=mask,
                                          sigma_rec=sigma_rec,
                                          sigma_inp=sigma_inp)
mse_score = np.round(mse_score, 7)
print(f"MSE: {mse_score}")

# Total variance
batch_size = input_batch_valid.shape[2]
RNN.clear_history()
circuit.clear_history()
RNN_trajectories, RNN_output = RNN.run_multiple_trajectories(input_timeseries=input_batch_valid, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
lc_trajectories, lc_output = circuit.run_multiple_trajectories(input_timeseries=input_batch_valid, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
lc_trajectories_emb = np.swapaxes(Q.T @ np.swapaxes(lc_trajectories, 0, 1), 0, 1)
RNN_trajectories_proj = np.swapaxes(Q @ np.swapaxes(RNN_trajectories, 0, 1), 0, 1)
r2_tot = np.mean([R2(lc_trajectories_emb[:, mask, i], RNN_trajectories[:, mask, i]) for i in range(batch_size)])
r2_proj = np.mean([R2(lc_trajectories[:, mask, i], RNN_trajectories_proj[:, mask, i]) for i in range(batch_size)])
print(f"Total R2: {r2_tot}")
print(f"Projected R2: {r2_proj}")
scores = {"mse_score": mse_score, "r2_tot": r2_tot, "r2_proj" : r2_proj}

SLURM_JOB_ID = None
data_folder = os.path.join(LCI_config_file["data_folder"], top_RNN + '_' + str(mse_score) + ('' if (SLURM_JOB_ID is None) else str(SLURM_JOB_ID)))
datasaver = DataSaver(data_folder)

datasaver.save_data(scores, f"LC_scores_{mse_score}.pkl")
datasaver.save_data(LCI_config_file, f"LC_config_{mse_score}.json")
datasaver.save_data(net_params, f"LC_params_{mse_score}.pkl")

w_rec = net_params["W_rec"]
fig_w_rec = analyzer.plot_recurrent_matrix()
datasaver.save_figure(fig_w_rec, f"LC_wrec_{mse_score}.png")
plt.show()

fig_w_rec_comparison = analyzer.plot_recurrent_matrix_comparison(w_rec_bar=w_rec_bar)
datasaver.save_figure(fig_w_rec_comparison, f"LC_wrec_comparison_{mse_score}.png")
plt.show()

fig_circuit_graph = analyzer.plot_circuit_graph()
datasaver.save_figure(fig_circuit_graph, f"circuit_graph_{mse_score}.png")
plt.show()

print(f"Plotting random trials")
inds = np.random.choice(np.arange(input_batch_valid.shape[-1]), 12)
inputs = input_batch_valid[..., inds]
targets = target_batch_valid[..., inds]
fig_trials = analyzer.plot_trials(inputs, targets, mask, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
datasaver.save_figure(fig_trials, f"LC_random_trials_{r2_tot}.png")
plt.show()

num_levels = len(task_data["task_params"][0]["coherences"])
analyzer.calc_psychometric_data(task, mask, num_levels=num_levels, num_repeats=31, sigma_rec=0.03, sigma_inp=0.03)
fig_psycho = analyzer.plot_psychometric_data()
datasaver.save_figure(fig_psycho, f"LC_psychometric_{r2_tot}.png")
datasaver.save_data(analyzer.psychometric_data, f"{r2_tot}_psycho_data.pkl")
plt.show()

print(f"Analyzing fixed points")
dsa = DynamicSystemAnalyzerCDDM(circuit)
params = {"fun_tol": 0.05,
          "diff_cutoff": 1e-4,
          "sigma_init_guess": 15,
          "patience": 100,
          "stop_length": 100,
          "mode": "approx"}
dsa.get_fixed_points(Input=np.array([1, 0, 0.5, 0.5, 0.5, 0.5]), **params)
dsa.get_fixed_points(Input=np.array([0, 1, 0.5, 0.5, 0.5, 0.5]), **params)
print(f"Calculating Line Attractor analytics")
dsa.calc_LineAttractor_analytics()
fig_LA3D = dsa.plot_LineAttractor_3D()
datasaver.save_figure(fig_LA3D, f"LC_LA3D_{r2_tot}.png")
datasaver.save_data(dsa.fp_data, f"{r2_tot}_fp_data.pkl")
datasaver.save_data(dsa.LA_data, f"{r2_tot}_LA_data.pkl")
plt.show()