import sys
from copy import deepcopy
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from pathlib import Path
projects_folder = str(Path.home()) + "/Documents/GitHub/"
sys.path.append(projects_folder)
import numpy as np
import json
import os
import torch
from rnn_coach.src.RNN_torch import *
from rnn_coach.src.RNN_numpy import *
from rnn_coach.src.Task import *
from rnn_coach.src.DataSaver import *
from src.utils import jsonify
from latent_circuit_inference.src.LatentCircuit import *
from latent_circuit_inference.src.LatentCircuitFitter import *
from latent_circuit_inference.src.LCAnalyzer import *
from latent_circuit_inference.src.utils import *
from matplotlib import pyplot as plt

from pathlib import Path
home = str(Path.home())
if home == '/home/pt1290':
    projects_folder = home
    data_save_path = home + '/latent_circuit_inference/data/inferred_LCs'
    LC_configs_path = home + '/latent_circuit_inference/data/configs'
elif home == '/Users/tolmach':
    projects_folder = home + '/Documents/GitHub/'
    data_save_path = projects_folder + '/latent_circuit_inference/data/inferred_LCs'
    LC_configs_path = projects_folder + '/latent_circuit_inference/data/configs'
else:
    pass

def mse_scoring(x, y):
    return np.mean((x - y) ** 2)

def R2(x, y):
    return 1.0 - mse_scoring(x, y)/np.var(y)

print(f"system arguments: {sys.argv}")
print(f"system arguments: {sys.argv}")
# RNN_folder = sys.argv[1]
RNN_folder = '0.0011968_BlockDMtanh;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000'
tag = 'BlockDMtanh'
taskname = RNN_folder.split("_")[1].split(";")[0]
data_save_path = os.path.join(data_save_path, taskname)
disp = False

RNN_folder_full_path = os.path.join(projects_folder, "rnn_coach", "data", "trained_RNNs", taskname, RNN_folder)
mse_score_RNN = RNN_folder.split("_")[0]

rnn_config = json.load(open(os.path.join(RNN_folder_full_path, f"{mse_score_RNN}_config.json"), "rb+"))
rnn_data = json.load(open(os.path.join(RNN_folder_full_path, f"{mse_score_RNN}_params_{taskname}.json"), "rb+"))
LCI_config_file = json.load(open(os.path.join(LC_configs_path, f"LCI_config_{tag}.json"), mode="r", encoding="utf-8"))
task_data = rnn_config["task_params"]
tmp = task_data["coherences"][-1] * np.logspace(-(7 - 1), 0, 7, base=2)
coherences = np.concatenate([-np.array(tmp[::-1]), np.array([0]), np.array(tmp)]).tolist()

task_data["coherences"] = deepcopy(coherences)

for N_LC in [8, 9]:
    for trial in range(15):
        print(RNN_folder, trial)
        # defining RNN:
        activation_name_RNN = rnn_config["activation"]
        RNN_N = rnn_config["N"]
        match activation_name_RNN:
            case "relu": activation_RNN = lambda x: torch.maximum(x, torch.tensor(0))
            case "tanh": activation_RNN = lambda x: torch.tanh(x)
            case "sigmoid": activation_RNN = lambda x: 1/(1 + torch.exp(-x))
            case "softplus": activation_RNN = lambda x: torch.log(1 + torch.exp(5 * x))

        dt = rnn_config["dt"]
        tau = rnn_config["tau"]
        connectivity_density_rec = rnn_config["connectivity_density_rec"]
        spectral_rad = rnn_config["sr"]
        sigma_inp = rnn_config["sigma_inp"]
        sigma_rec = rnn_config["sigma_rec"]
        seed = np.random.randint(1000000)
        print(f"seed: {seed}")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        rng = torch.Generator(device=torch.device(device))
        if not seed is None:
            rng.manual_seed(seed)
        input_size = np.array(rnn_data["W_inp"]).shape[1]
        output_size = np.array(rnn_data["W_out"]).shape[0]

        # Task:
        n_steps = task_data["n_steps"]

        N = N_LC # LCI_config_file["N"]
        N_PCs = LCI_config_file["N_PCs"]
        w_inp = np.zeros((N, input_size))
        w_inp[:input_size, :input_size] = np.eye(input_size)
        w_out = np.zeros((output_size, N))
        w_out[:, input_size:] = 1

        # Fitter:
        lambda_w = LCI_config_file["lambda_w"]
        max_iter = LCI_config_file["max_iter"]
        tol = LCI_config_file["tol"]
        lr = LCI_config_file["lr"]
        actvation_name_LC = LCI_config_file["activation"]
        inp_connectivity_mask = np.array(LCI_config_file["inp_connectivity_mask"])
        rec_connectivity_mask = np.array(LCI_config_file["rec_connectivity_mask"])
        out_connectivity_mask = np.array(LCI_config_file["out_connectivity_mask"])
        Qinitialization = LCI_config_file["Qinitialization"]
        encoding = LCI_config_file["encoding"]

        match actvation_name_LC:
            case "relu": activation_LC = lambda x: torch.maximum(x, torch.tensor(0))
            case "tanh": activation_LC = lambda x: torch.tanh(x)
            case "sigmoid": activation_LC = lambda x: 1/(1 + torch.exp(-x))
            case "softplus": activation_LC = lambda x: torch.log(1 + torch.exp(5 * x))

        # # creating instances:
        rnn_torch = RNN_torch(N=RNN_N, dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                              activation=activation_RNN, random_generator=rng, device=device,
                              sigma_rec=sigma_rec, sigma_inp=sigma_inp)
        RNN_params = {"W_inp": np.array(rnn_data["W_inp"]),
                      "W_rec": np.array(rnn_data["W_rec"]),
                      "W_out": np.array(rnn_data["W_out"]),
                      "b_rec": np.array(rnn_data["bias_rec"]),
                      "y_init": np.zeros(RNN_N)}
        rnn_torch.set_params(RNN_params)
        task = TaskBlockDMtanh(n_steps=n_steps, task_params=task_data)

        lc = LatentCircuit(N=N,
                           W_inp=torch.Tensor(w_inp).to(device),
                           W_out=torch.Tensor(w_out).to(device),
                           num_inputs=input_size,
                           num_outputs=output_size,
                           inp_connectivity_mask=torch.Tensor(w_inp).to(device),
                           rec_connectivity_mask=torch.Tensor(np.ones((N, N))).to(device),
                           out_connectivity_mask=torch.Tensor(w_out).to(device),
                           activation=activation_LC,
                           sigma_rec=sigma_rec,
                           sigma_inp=sigma_inp,
                           device=device,
                           random_generator=rng)
        criterion = torch.nn.MSELoss()
        fitter = LatentCircuitFitter(LatentCircuit=lc, RNN=rnn_torch, Task=task,
                                     N_PCs = N_PCs,
                                     encoding = encoding,
                                     max_iter=max_iter, tol=tol, lr = lr,
                                     criterion=criterion,
                                     lambda_w=lambda_w,
                                     Qinitialization=Qinitialization)

        lc_inferred, train_losses, val_losses, net_params = fitter.run_training()
        # defining circuit
        # N_LC = LCI_config_file["N"]
        U = net_params["U"]
        q = net_params["q"]
        Q = U.T @ q
        W_rec = RNN_params["W_rec"]
        w_rec_bar = Q.T @ W_rec @ Q
        w_rec = net_params["W_rec"]
        names = ["ctx m", "ctx c", "mr", "ml", "cr", "cl", "OutR", "OutL"]
        w_rec = net_params["W_rec"]
        w_inp = net_params["W_inp"]
        w_out = net_params["W_out"]
        dt = net_params["dt"]
        tau = net_params["tau"]

        activation_fun_circuit = lambda x: np.tanh(x)
        circuit = RNN_numpy(N=N_LC, W_rec=w_rec, W_inp=w_inp, W_out=w_out, dt=dt, tau=tau, activation=activation_fun_circuit)
        circuit.y = np.zeros(N_LC)

        # defining RNN
        N_RNN = rnn_data["N"]
        x = np.random.randn(N_RNN)
        W_rec = RNN_params["W_rec"]
        W_inp = RNN_params["W_inp"]
        W_out = RNN_params["W_out"]
        dt = net_params["dt"]
        tau = net_params["tau"]
        match activation_name_RNN:
            case "relu": activation_fun_RNN_np = lambda x: np.maximum(x, 0)
            case "tanh": activation_fun_RNN_np = lambda x: np.tanh(x)
            case "sigmoid": activation_fun_RNN_np = lambda x: 1 / (1 + np.exp(-x))
            case "softplus": activation_fun_RNN_np = lambda x: np.log(1 + np.exp(5 * x))
        RNN = RNN_numpy(N=N_RNN, W_rec=W_rec, W_inp=W_inp, W_out=W_out, dt=dt, tau=tau, activation=activation_fun_RNN_np)
        RNN.y = np.zeros(N_RNN)

        node_labels = np.arange(w_rec.shape[0])
        analyzer = LCAnalyzer(circuit, labels=node_labels)
        input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
        mask = np.array(rnn_config["mask"])

        #MSE mse_score_RNN
        score_function = lambda x, y: np.mean((x - y) ** 2)
        mse_score = analyzer.get_validation_score(scoring_function=mse_scoring,
                                                  input_batch=input_batch_valid,
                                                  target_batch=target_batch_valid,
                                                  mask=mask,
                                                  sigma_rec=sigma_rec,
                                                  sigma_inp=sigma_inp)
        mse_score = np.round(mse_score, 8)
        print(f"MSE: {mse_score}")

        # Total variance
        batch_size = input_batch_valid.shape[2]
        RNN.clear_history()
        circuit.clear_history()
        RNN.run(input_timeseries=input_batch_valid, sigma_rec=0, sigma_inp=0)
        RNN_trajectories = RNN.get_history()
        RNN_output = RNN.get_output()
        circuit.run(input_timeseries=input_batch_valid, sigma_rec=0, sigma_inp=0)
        lc_trajectories = circuit.get_history()
        lc_output = circuit.get_output()

        lc_trajectories_emb = np.einsum("ji, ikp->jkp", Q, lc_trajectories)
        RNN_trajectories_proj = np.einsum("ij, ikp->jkp", Q, RNN_trajectories)
        r2_tot = np.nanmean([R2(lc_trajectories_emb[:, mask, i], RNN_trajectories[:, mask, i]) for i in range(batch_size)])
        r2_proj = np.nanmean([R2(lc_trajectories[:, mask, i], RNN_trajectories_proj[:, mask, i]) for i in range(batch_size)])
        print(f"Total R2: {r2_tot}")
        print(f"Projected R2: {r2_proj}")
        scores = {"mse_score": mse_score, "r2_tot":r2_tot, "r2_proj" : r2_proj}

        # if r2_tot > 0.82:
        data_save_folder = os.path.join(data_save_path, RNN_folder, f"{N_LC}_{r2_tot}_{r2_proj}_LC_{tag}")
        datasaver = DataSaver(data_save_folder)
        datasaver.save_data(scores, f"{r2_tot}_{r2_proj}_LC_scores.json")
        datasaver.save_data(jsonify(LCI_config_file), f"{r2_tot}_{r2_proj}_LC_config.json")
        datasaver.save_data(jsonify(net_params), f"{r2_tot}_{r2_proj}_LC_params.json")
        # saving RNN data alongside
        datasaver.save_data(jsonify(rnn_config), f"{mse_score_RNN}_config.json")
        datasaver.save_data(jsonify(rnn_data), f"{mse_score_RNN}_params_{taskname}.json")


        w_rec = net_params["W_rec"]
        fig_w_rec = analyzer.plot_recurrent_matrix()
        datasaver.save_figure(fig_w_rec, f"{r2_tot}_{r2_proj}_LC_wrec.png")
        if disp: plt.show()

        w_rec = net_params["W_rec"]
        labels = np.arange(N_LC).tolist()
        labels[0] = 'block'
        labels[1] = 'inp'
        fig_circuit, ax = analyzer.plot_circuit(labels)
        datasaver.save_figure(fig_circuit, f"{r2_tot}_{r2_proj}_LC_circuit.png")
        if disp: plt.show()

        w_out = net_params["W_out"]
        fig_w_out = analyzer.plot_output_matrix()
        datasaver.save_figure(fig_w_out, f"{r2_tot}_{r2_proj}_LC_wout.png")
        if disp: plt.show()

        fig_w_rec_comparison = analyzer.plot_recurrent_matrix_comparison(w_rec_bar=w_rec_bar)
        datasaver.save_figure(fig_w_rec_comparison, f"{r2_tot}_{r2_proj}_LC_wrec_comparison.png")
        if disp: plt.show()

        print(f"Plotting random trials")
        inds = np.random.choice(np.arange(input_batch_valid.shape[-1]), 12)
        inputs = input_batch_valid[..., inds]
        targets = target_batch_valid[..., inds]
        fig_trials = analyzer.plot_trials(inputs, targets, mask, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
        datasaver.save_figure(fig_trials, f"{r2_tot}_{r2_proj}_LC_random_trials.png")
        if disp: plt.show()
