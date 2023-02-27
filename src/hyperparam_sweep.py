import optuna_distributed
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_slice, plot_contour
import warnings
import optuna
import sys
sys.path.append("../../")
import numpy as np
from matplotlib import pyplot as plt
import json
import pickle
import os
import torch
from rnn_coach.src.RNN_torch import *
from rnn_coach.src.DynamicSystemAnalyzer import *
from rnn_coach.src.RNN_numpy import *
from rnn_coach.src.Task import *
from rnn_coach.src.DataSaver import *
from latent_circuit_inference.src.LatentCircuit import *
from latent_circuit_inference.src.LatentCircuitFitter import *
from latent_circuit_inference.src.utils import *
from latent_circuit_inference.src.circuit_vizualization import *
warnings.simplefilter("ignore", UserWarning)
import time

def mse_scoring(x, y):
    return np.mean((x - y) ** 2)

def R2(x, y):
    return 1.0 - mse_scoring(x, y)/np.var(y)

def objective(trial):
    r2_tot_list = []
    r2_proj_list = []
    lambda_w = trial.suggest_loguniform('lambda_w', 0.01, 0.3)
    lr = trial.suggest_loguniform('lr', 0.01, 0.3)
    for seed in [0, 1, 2, 3, 4]:
        # sigma_rec = trial.suggest_loguniform('sigma_rec', 0.001, 0.04)
        sigma_inp = 0
        RNN_folder = "20230207-08111997"
        tag = '8-nodes'
        RNN_folder_full_path = os.path.join("../", "../", "rnn_coach", "data", "trained_RNNs", "CDDM", RNN_folder)
        score = os.listdir(RNN_folder_full_path)[0].split("_")[0]
        rnn_config = json.load(open(os.path.join(RNN_folder_full_path, f"{score}_config.json"), "rb+"))
        rnn_data = pickle.load(open(os.path.join(RNN_folder_full_path, f"{score}_params_CDDM.pkl"), "rb+"))
        LCI_config_file = json.load(
            open(os.path.join("../", "data", "configs", f"LCI_config.json"), mode="r", encoding='utf-8'))
        task_data = rnn_config["task_params"]

        # defining RNN:
        activation_name = rnn_config["activation"]
        RNN_N = rnn_config["N"]
        if activation_name == 'relu':
            activation_RNN = lambda x: torch.maximum(x, torch.tensor(0))
        elif activation_name == 'tanh':
            activation_RNN = torch.tanh
        elif activation_name == 'sigmoid':
            activation_RNN = lambda x: 1 / (1 + torch.exp(-x))
        elif activation_name == 'softplus':
            activation_RNN = lambda x: torch.log(1 + torch.exp(5 * x))
        dt = rnn_config["dt"]
        tau = rnn_config["tau"]
        sigma_inp = rnn_config["sigma_inp"]
        sigma_rec = rnn_config["sigma_rec"]

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        rng = torch.Generator(device=torch.device('cpu'))
        if not seed is None:
            rng.manual_seed(seed)
        input_size = np.array(rnn_data["W_inp"]).shape[1]
        output_size = np.array(rnn_data["W_out"]).shape[0]

        # Task:
        n_steps = task_data["n_steps"]

        # LC
        n = LCI_config_file["n"]
        LC_N = LCI_config_file["N"]
        W_inp = np.array(LCI_config_file["W_inp"])
        W_out = np.array(LCI_config_file["W_out"])
        # Fitter:
        # lambda_w = LCI_config_file["lambda_w"]
        max_iter = LCI_config_file["max_iter"]
        tol = LCI_config_file["tol"]
        # lr = LCI_config_file["lr"]
        actvation_name = LCI_config_file["activation"]

        inp_connectivity_mask = np.array(LCI_config_file["inp_connectivity_mask"])
        rec_connectivity_mask = np.array(LCI_config_file["rec_connectivity_mask"])
        out_connectivity_mask = np.array(LCI_config_file["out_connectivity_mask"])
        if activation_name == 'relu':
            activation_LC = lambda x: torch.maximum(x, torch.tensor(0))
        elif activation_name == 'tanh':
            activation_LC = torch.tanh
        elif activation_name == 'sigmoid':
            activation_LC = lambda x: 1 / (1 + torch.exp(-x))
        elif activation_name == 'softplus':
            activation_LC = lambda x: torch.log(1 + torch.exp(5 * x))

        # # creating instances:
        rnn_torch = RNN_torch(N=RNN_N, dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                              activation=activation_RNN, random_generator=rng, device=device)
        RNN_params = {"W_inp": np.array(rnn_data["W_inp"]),
                      "W_rec": np.array(rnn_data["W_rec"]),
                      "W_out": np.array(rnn_data["W_out"]),
                      "b_rec": np.array(rnn_data["bias_rec"]),
                      "y_init": np.zeros(RNN_N)}
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

        task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_data)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(lc.parameters(),
                                     lr=lr)
        fitter = LatentCircuitFitter(LatentCircuit=lc, RNN=rnn_torch, Task=task,
                                     max_iter=max_iter, tol=tol,
                                     optimizer=optimizer, criterion=criterion,
                                     lambda_w=lambda_w)
        try:
            tic = time.perf_counter()
            lc_inferred, train_losses, val_losses, net_params = fitter.run_training()
            toc = time.perf_counter()
            print(f"Executed training in {toc - tic:0.4f} seconds")
        except:
            r2_tot_list.append(-100)
            r2_proj_list.append(-100)
            print(f"Failed to converge, seed : {seed}")
            break
        # defining circuit
        n = 8
        U = net_params["U"]
        q = net_params["q"]
        Q = q @ U  # should be 100 x 8
        W_rec = RNN_params["W_rec"]
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
        W_rec = RNN_params["W_rec"]
        W_inp = RNN_params["W_inp"]
        W_out = RNN_params["W_out"]
        dt = net_params["dt"]
        tau = net_params["tau"]
        activation_fun_RNN = lambda x: np.maximum(0, x)
        RNN = RNN_numpy(N=N, W_rec=W_rec, W_inp=W_inp, W_out=W_out, dt=dt, tau=tau, activation=activation_fun_RNN)
        RNN.y = np.zeros(n)

        input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
        mask = np.array(rnn_config["mask"])

        # Total variance
        batch_size = input_batch_valid.shape[2]
        RNN.clear_history()
        circuit.clear_history()
        RNN_trajectories, RNN_output = RNN.run_multiple_trajectories(input_timeseries=input_batch_valid,
                                                                     sigma_rec=sigma_rec, sigma_inp=sigma_inp)
        lc_trajectories, lc_output = circuit.run_multiple_trajectories(input_timeseries=input_batch_valid,
                                                                       sigma_rec=sigma_rec, sigma_inp=sigma_inp)
        lc_trajectories_emb = np.swapaxes(Q.T @ np.swapaxes(lc_trajectories, 0, 1), 0, 1)
        RNN_trajectories_proj = np.swapaxes(Q @ np.swapaxes(RNN_trajectories, 0, 1), 0, 1)
        r2_tot = np.mean([R2(lc_trajectories_emb[:, mask, i], RNN_trajectories[:, mask, i]) for i in range(batch_size)])
        r2_proj = np.mean([R2(lc_trajectories[:, mask, i], RNN_trajectories_proj[:, mask, i]) for i in range(batch_size)])
        r2_tot_list.append(r2_tot)
        r2_proj_list.append(r2_proj)
        print(f"seed : {seed}, r2_tot : {r2_tot}")
    print(f"mean r2_tot = {np.mean(r2_tot_list)}")
    print(f"mean r2_proj = {np.mean(r2_proj_list)}")
    return np.mean(r2_tot_list)

if __name__ == '__main__':
    client = None  # Enables local asynchronous optimization.
    # study = optuna_distributed.from_study(optuna.create_study(direction="maximize"), client=client)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=41)
    print(f"best parameters : {study.best_params}")
    fig = plot_optimization_history(study).get_figure()
    fig.savefig("../img/optimization_history_mse.pdf")
    fig = plot_param_importances(study).get_figure()
    fig.savefig("../img/param_importances_mse.pdf")
    fig = plot_slice(study, params=["lambda_w"]).get_figure()
    fig.savefig("../img/slice_plot_mse.pdf").get_figure()
    fig = plot_contour(study, params=["lr", "lambda_w"]).get_figure()
    fig.savefig("../img/contour_plot_mse.pdf")