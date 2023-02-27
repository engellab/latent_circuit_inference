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
from latent_circuit_inference.src.LCAnalyzer import *
from latent_circuit_inference.src.utils import *
from latent_circuit_inference.src.circuit_vizualization import *

def mse_scoring(x, y):
    return np.mean((x - y) ** 2)

def R2(x, y):
    return 1.0 - mse_scoring(x, y)/np.var(y)

#given the folder, open up the files:
RNN_folder = sys.argv[1]
# RNN_folder = '0.0070184_20230222-083339'
tag = '8-nodes'
RNN_folder_full_path = os.path.join("../", "../", "rnn_coach", "data", "trained_RNNs", "CDDM", RNN_folder)
mse_score_RNN = os.listdir(RNN_folder_full_path)[0].split("_")[0]
rnn_config = json.load(open(os.path.join(RNN_folder_full_path, f"{mse_score_RNN}_config.json"), "rb+"))
# try:
#     rnn_data = pickle.load(open(os.path.join(RNN_folder_full_path, f"{mse_score_RNN}_params_CDDM.pkl"), "rb+"))
# except:
#     rnn_data = pickle.load(open(os.path.join(RNN_folder_full_path, f"params_CDDM_{mse_score_RNN}.pkl"), "rb+"))
rnn_data = json.load(open(os.path.join(RNN_folder_full_path, f"{mse_score_RNN}_params_CDDM.json"), "rb+"))
LCI_config_file = json.load(open(os.path.join("../", "data", "configs", f"LCI_config.json"), mode="r", encoding='utf-8'))
task_data = rnn_config["task_params"]

for trial in range(30):
    # defining RNN:
    activation_name = rnn_config["activation"]
    RNN_N = rnn_config["N"]

    if activation_name == 'relu':
        activation_RNN = lambda x: torch.maximum(x, torch.tensor(0))
    elif activation_name == 'tanh':
        activation_RNN = torch.tanh
    elif activation_name == 'sigmoid':
        activation_RNN = lambda x: 1/(1 + torch.exp(-x))
    elif activation_name == 'softplus':
        activation_RNN = lambda x: torch.log(1 + torch.exp(5 * x))
    dt = rnn_config["dt"]
    tau = rnn_config["tau"]
    connectivity_density_rec = rnn_config["connectivity_density_rec"]
    spectral_rad = rnn_config["sr"]
    sigma_inp = rnn_config["sigma_inp"]
    sigma_rec = rnn_config["sigma_rec"]
    seed = np.random.randint(1000000)
    print(f"seed: {seed}")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    rng = torch.Generator(device=torch.device(device))
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
                          activation=activation_RNN, random_generator=rng, device=device,
                          sigma_rec=sigma_rec, sigma_inp=sigma_inp)
    RNN_params = {"W_inp": np.array(rnn_data["W_inp"]),
                  "W_rec": np.array(rnn_data["W_rec"]),
                  "W_out": np.array(rnn_data["W_out"]),
                  "b_rec": np.array(rnn_data["bias_rec"]),
                  "y_init": np.zeros(RNN_N)}
    rnn_torch.set_params(RNN_params)

    lc = LatentCircuit(n=n,
                       N=LC_N,
                       W_inp=torch.Tensor(W_inp).to(device),
                       W_out=torch.Tensor(W_out).to(device),
                       inp_connectivity_mask=torch.Tensor(inp_connectivity_mask).to(device),
                       rec_connectivity_mask=torch.Tensor(rec_connectivity_mask).to(device),
                       out_connectivity_mask=torch.Tensor(out_connectivity_mask).to(device),
                       activation=activation_LC,
                       sigma_rec=sigma_rec,
                       sigma_inp=sigma_inp,
                       device=device,
                       random_generator=rng)

    task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_data)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lc.parameters(), lr=lr)
    fitter = LatentCircuitFitter(LatentCircuit=lc, RNN=rnn_torch, Task=task,
                                 max_iter=max_iter, tol=tol,
                                 optimizer=optimizer, criterion=criterion,
                                 lambda_w=lambda_w)

    lc_inferred, train_losses, val_losses, net_params = fitter.run_training()

    # defining circuit
    n = LCI_config_file["n"]
    U = net_params["U"]
    q = net_params["q"]
    Q = q @ U
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
    N = rnn_data["N"]
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
    RNN_trajectories, RNN_output = RNN.run_multiple_trajectories(input_timeseries=input_batch_valid, sigma_rec=0, sigma_inp=0)
    lc_trajectories, lc_output = circuit.run_multiple_trajectories(input_timeseries=input_batch_valid, sigma_rec=0, sigma_inp=0)
    lc_trajectories_emb = np.swapaxes(Q.T @ np.swapaxes(lc_trajectories, 0, 1), 0, 1)
    RNN_trajectories_proj = np.swapaxes(Q @ np.swapaxes(RNN_trajectories, 0, 1), 0, 1)
    r2_tot = np.mean([R2(lc_trajectories_emb[:, mask, i], RNN_trajectories[:, mask, i]) for i in range(batch_size)])
    r2_proj = np.mean([R2(lc_trajectories[:, mask, i], RNN_trajectories_proj[:, mask, i]) for i in range(batch_size)])
    print(f"Total R2: {r2_tot}")
    print(f"Projected R2: {r2_proj}")
    scores = {"mse_score": mse_score, "r2_tot":r2_tot, "r2_proj" : r2_proj}

    data_folder = os.path.join(LCI_config_file["data_folder"], RNN_folder, f"{r2_tot}_LC_{tag}")
    datasaver = DataSaver(data_folder)
    datasaver.save_data(scores, f"{r2_tot}_LC_scores.json")
    datasaver.save_data(LCI_config_file, f"{r2_tot}_LC_config.json")
    datasaver.save_data(net_params, f"{r2_tot}_LC_params.pkl")

    w_rec = net_params["W_rec"]
    fig_w_rec = analyzer.plot_recurrent_matrix()
    datasaver.save_figure(fig_w_rec, f"{r2_tot}_LC_wrec.png")
    # plt.show()

    fig_w_rec_comparison = analyzer.plot_recurrent_matrix_comparison(w_rec_bar=w_rec_bar)
    datasaver.save_figure(fig_w_rec_comparison, f"{r2_tot}_LC_wrec_comparison.png")
    # plt.show()

    fig_circuit_graph = analyzer.plot_circuit_graph()
    datasaver.save_figure(fig_circuit_graph, f"{r2_tot}_circuit_graph.png")
    # plt.show()

    print(f"Plotting random trials")
    inds = np.random.choice(np.arange(input_batch_valid.shape[-1]), 12)
    inputs = input_batch_valid[..., inds]
    targets = target_batch_valid[..., inds]
    fig_trials = analyzer.plot_trials(inputs, targets, mask, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
    datasaver.save_figure(fig_trials, f"{r2_tot}_LC_random_trials.png")
    # plt.show()

    num_levels = len(task_data["coherences"])
    analyzer.calc_psychometric_data(task, mask, num_levels=num_levels, num_repeats=31, sigma_rec=0.03, sigma_inp=0.03)
    fig_psycho = analyzer.plot_psychometric_data()
    datasaver.save_figure(fig_psycho, f"{r2_tot}_LC_psychometric.png")
    datasaver.save_data(analyzer.psychometric_data, f"{r2_tot}_psycho_data.pkl")
    # plt.show()

    print(f"Analyzing fixed points")
    dsa = DynamicSystemAnalyzerCDDM(circuit)
    params = {"fun_tol": 0.05,
              "diff_cutoff": 1e-4,
              "sigma_init_guess": 15,
              "patience": 50,
              "stop_length": 50,
              "mode": "approx"}
    dsa.get_fixed_points(Input=np.array([1, 0, 0.5, 0.5, 0.5, 0.5]), **params)
    dsa.get_fixed_points(Input=np.array([0, 1, 0.5, 0.5, 0.5, 0.5]), **params)
    print(f"Calculating Line Attractor analytics")
    dsa.calc_LineAttractor_analytics()
    fig_LA3D = dsa.plot_LineAttractor_3D()
    datasaver.save_figure(fig_LA3D, f"{r2_tot}_LC_LA3D.png")
    datasaver.save_data(dsa.fp_data, f"{r2_tot}_fp_data.pkl")
    datasaver.save_data(dsa.LA_data, f"{r2_tot}_LA_data.pkl")
    # plt.show()

    LA_data_lc = pickle.load(open(os.path.join(data_folder, f"{r2_tot}_LA_data.pkl"), "rb+"))
    LA_data_RNN = pickle.load(open(os.path.join(RNN_folder_full_path, f"{mse_score_RNN}_LA_data.pkl"), "rb+"))
    fig_selection_vects = analyzer.plot_selection_vectors(Q, LA_data_lc, LA_data_RNN)
    datasaver.save_figure(fig_selection_vects, f"{r2_tot}_selection_vects_comparison.png")
    # plt.show()