import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)
import os
import sys
import pickle
import json
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
sys.path.insert(0, "../../../")
from matplotlib import image as mpimg
import torch
from scipy.stats import zscore
from rnn_coach.src.Task import TaskCDDM
from rnn_coach.src.RNN_torch import RNN_torch
from rnn_coach.src.RNN_numpy import RNN_numpy
from scipy.sparse.linalg import lsqr
from copy import deepcopy
from sklearn.decomposition import PCA
from latent_circuit_inference.DataSaver import DataSaver
from latent_circuit_inference.DynamicSystemAnalyzer import *
from latent_circuit_inference.PerformanceAnalyzer import *
from latent_circuit_inference.RNN_numpy import RNN_numpy
from latent_circuit_inference.utils import get_project_root, numpify, orthonormalize
from latent_circuit_inference.Trainer import Trainer
from latent_circuit_inference.RNN_torch import RNN_torch
from latent_circuit_inference.PerformanceAnalyzer import PerformanceAnalyzerCDDM
import pickle

RNNs = []
for folder in os.listdir(os.path.join('../', '../', "data", "inferred_LCs")):
    if (folder == "exemplar_RNN") or (folder == '.DS_Store') or (folder == '0.0077001_20230222-010124'):
        pass
    else:
        if "CDDM;relu" in folder:
            RNNs.append(folder)
epsilons = [1.42, 1.44, 1.46, 1.48, 1.5]
RNNs = ["0.0069969_CDDM;relu;N=92;lmbdo=0.3;lmbdr=0.0;lr=0.002;maxiter=3000"]
for epsilon in epsilons:
    for RNN_folder in RNNs:
        RNN_score = float(RNN_folder.split("_")[0])
        RNN_path = os.path.join('../', '../', '../', "rnn_coach", "data", "trained_RNNs", "CDDM", RNN_folder)
        LC_folder_path = os.path.join('../', '../', "data", "inferred_LCs", RNN_folder)
        subfolders = os.listdir(LC_folder_path)
        ind = 0
        max_score = -10
        for i, subfolder in enumerate(subfolders):
            if "8nodes" in subfolder or '8-nodes' in subfolder:
                score = float(subfolder.split("_")[0])
                if max_score <= score:
                    max_score = score
                    ind = i
        LC_subfolder = subfolders[ind]
        LC_path = os.path.join('../', '../', "data", "inferred_LCs", RNN_folder, LC_subfolder)

        if max_score > 0.90:
            print(RNN_folder)
            rnn_config = json.load(open(os.path.join(RNN_path, f"{RNN_score}_config.json"), "rb+"))
            rnn_data = json.load(open(os.path.join(RNN_path, f"{RNN_score}_params_CDDM.json"), "rb+"))
            train_config_file = f"train_config_CDDM_relu.json"
            activation_name = rnn_config["activation"]
            RNN_N = rnn_config["N"]
            n_steps = rnn_config["n_steps"]
            task_params = rnn_config["task_params"]
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
            mask = np.array(rnn_config["mask"])
            RNN = RNN_torch(N=RNN_N, dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                            activation=activation_RNN, random_generator=rng, device=device,
                            sigma_rec=sigma_rec, sigma_inp=sigma_inp)
            RNN_params = {"W_inp": np.array(rnn_data["W_inp"]),
                          "W_rec": np.array(rnn_data["W_rec"]),
                          "W_out": np.array(rnn_data["W_out"]),
                          "b_rec": np.array(rnn_data["bias_rec"]),
                          "y_init": np.zeros(RNN_N)}
            RNN.set_params(RNN_params)
            task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
            # validate
            coherences_valid = np.linspace(-1, 1, 11)
            task_params_valid = deepcopy(task_params)
            task_params_valid["coherences"] = coherences_valid
            task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params_valid)

            RNN_valid_control = RNN_numpy(N=RNN_N,
                                          dt=rnn_data["dt"],
                                          tau=rnn_data["tau"],
                                          activation=lambda x: np.maximum(0.0, x),
                                          W_inp=np.array(rnn_data["W_inp"]),
                                          W_rec=np.array(rnn_data["W_rec"]),
                                          W_out=np.array(rnn_data["W_out"]),
                                          bias_rec=None,
                                          y_init=np.zeros(RNN_N))

            analyzer = PerformanceAnalyzerCDDM(RNN_valid_control)
            score_function = lambda x, y: np.mean((x - y) ** 2)
            Task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)

            input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
            score = analyzer.get_validation_score(score_function, input_batch_valid, target_batch_valid, mask, sigma_rec=0,
                                                  sigma_inp=0.)
            score = np.round(score, 7)

            try:
                LC_data = json.load(open(os.path.join(LC_path, f"{max_score}_LC_params.json"), "rb+"))
            except:
                LC_data = pickle.load(open(os.path.join(LC_path, f"{max_score}_LC_params.pkl"), "rb+"))
            U = np.array(LC_data["U"])
            q = np.array(LC_data["q"])
            try:
                Q = U.T @ q
            except:
                Q = (q @ U).T
            w_out = np.array(LC_data["W_out"])
            w_rec = np.array(LC_data["W_rec"])
            w_inp = np.array(LC_data["W_inp"])
            n = LC_data["n"]
            N = LC_data["N"]
            dt = LC_data["dt"]
            tau = LC_data["tau"]

            # if there are files already, load them:
            if os.path.isfile(os.path.join(LC_path, f"{score}_LA_data.pkl")):
                LA_data_control = pickle.load(open(os.path.join(RNN_path, f"{score}_LA_data.pkl"), "rb+"))
            # if os.path.isfile(os.path.join(LC_path, f"{max_score}_RNN_LA_data.pkl")):
            #     LA_data_control = pickle.load(open(os.path.join(RNN_path, f"{max_score}_RNN_LA_data.pkl"), "rb+"))
            else:
                datasaver = DataSaver(LC_path)
                dsa_control = DynamicSystemAnalyzerCDDM(RNN_valid_control)
                params = {"fun_tol": 0.05,
                          "diff_cutoff": 1e-4,
                          "sigma_init_guess": 5,
                          "patience": 10,
                          "stop_length": 50,
                          "mode": "approx"}
                # dsa_control.get_fixed_points(Input=np.array([1, 0, 0.5, 0.5, 0.5, 0.5]), **params)
                # dsa_control.get_fixed_points(Input=np.array([0, 1, 0.5, 0.5, 0.5, 0.5]), **params)
                print(f"Calculating Line Attractor analytics")
                dsa_control.calc_LineAttractor_analytics(obj_max_iter=20)
                datasaver.save_data(dsa_control.LA_data, f"{max_score}_RNN_LA_data.pkl")

            if os.path.isfile(os.path.join(LC_path, f"{score}_{epsilon}_LAdata_corrupted.pkl")):
                LA_data_corrupted = pickle.load(open(os.path.join(LC_path, f"{score}_{epsilon}_LAdata_corrupted.pkl"), "rb+"))
            else:
                datasaver = DataSaver(LC_path)
                # perturbation:
                delta_w = np.zeros((8, 8))
                delta_w[2, 1] = 1
                delta_w[3, 1] = 1
                delta_W = epsilon * Q @ delta_w @ Q.T  # add more excitation
                W_rec_corupted = np.array(rnn_data["W_rec"]) + delta_W

                RNN_valid_corrupted = RNN_numpy(N=RNN_N,
                                                dt=rnn_data["dt"],
                                                tau=rnn_data["tau"],
                                                activation=lambda x: np.maximum(0.0, x),
                                                W_inp=np.array(rnn_data["W_inp"]),
                                                W_rec=deepcopy(W_rec_corupted),
                                                W_out=np.array(rnn_data["W_out"]),
                                                bias_rec=None,
                                                y_init=np.zeros(RNN_N))
                dsa_corrupted = DynamicSystemAnalyzerCDDM(RNN_valid_corrupted)
                params = {"fun_tol": 0.05,
                          "diff_cutoff": 1e-4,
                          "sigma_init_guess": 5,
                          "patience": 10,
                          "stop_length": 50,
                          "mode": "approx"}
                # dsa_corrupted.get_fixed_points(Input=np.array([1, 0, 0.5, 0.5, 0.5, 0.5]), **params)
                # dsa_corrupted.get_fixed_points(Input=np.array([0, 1, 0.5, 0.5, 0.5, 0.5]), **params)
                print(f"Calculating Line Attractor analytics")
                dsa_corrupted.calc_LineAttractor_analytics(obj_max_iter=20)
                LA_data_corrupted = dsa_corrupted.LA_data
                datasaver.save_data(dsa_corrupted.LA_data, f"{score}_{epsilon}_LAdata_corrupted.pkl")

            if os.path.isfile(os.path.join(LC_path, f"{score}_{epsilon}_LAdata_enhanced.pkl")):
                LA_data_enhanced = pickle.load(open(os.path.join(LC_path, f"{score}_{epsilon}_LAdata_enhanced.pkl"), "rb+"))
            else:
                datasaver = DataSaver(LC_path)
                # perturbation:
                delta_w = np.zeros((8, 8))
                delta_w[2, 1] = 1
                delta_w[3, 1] = 1
                delta_W = -epsilon * Q @ delta_w @ Q.T  # add more inhibition
                W_rec_enhanced = np.array(rnn_data["W_rec"]) + delta_W

                RNN_valid_enhanced = RNN_numpy(N=RNN_N,
                                                dt=rnn_data["dt"],
                                                tau=rnn_data["tau"],
                                                activation=lambda x: np.maximum(0.0, x),
                                                W_inp=np.array(rnn_data["W_inp"]),
                                                W_rec=deepcopy(W_rec_enhanced),
                                                W_out=np.array(rnn_data["W_out"]),
                                                bias_rec=None,
                                                y_init=np.zeros(RNN_N))
                dsa_enhanced = DynamicSystemAnalyzerCDDM(RNN_valid_enhanced)
                params = {"fun_tol": 0.05,
                          "diff_cutoff": 1e-4,
                          "sigma_init_guess": 5,
                          "patience": 10,
                          "stop_length": 50,
                          "mode": "approx"}
                # dsa_enhanced.get_fixed_points(Input=np.array([1, 0, 0.5, 0.5, 0.5, 0.5]), **params)
                # dsa_enhanced.get_fixed_points(Input=np.array([0, 1, 0.5, 0.5, 0.5, 0.5]), **params)
                # print(f"Calculating Line Attractor analytics")
                dsa_enhanced.calc_LineAttractor_analytics(obj_max_iter=20)
                datasaver.save_data(dsa_enhanced.LA_data, f"{score}_{epsilon}_LAdata_enhanced.pkl")
                LA_data_enhanced = dsa_enhanced.LA_data

            # selection vector analysis
            SV_dict = {}
            for type in ["control", "corrupted", "enhanced"]:
                SV_dict[type] = {}
                for ctx in ["motion", "color"]:
                    s = eval(f"LA_data_{type}")[ctx]["l"]
                    # take only middle points and project them with Q
                    num_pts = s.shape[0]
                    s_filtered = s[int(np.floor(num_pts / 2)) - 5: int(np.ceil(num_pts // 2)) + 5, :]
                    s_projected = s_filtered @ Q
                    s_projected_mean = np.mean(s_projected, axis = 0)
                    s_projected_mean_normalised = s_projected_mean/np.linalg.norm(s_projected_mean)
                    SV_dict[type][ctx] = deepcopy(s_projected_mean_normalised)

            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            for i, ctx in enumerate(["motion", "color"]):
                ax[i].bar(np.arange(16)[::2] - 0.4, SV_dict["corrupted"][ctx], color='r', width=0.4, alpha=0.5, label='corrupted')
                ax[i].bar(np.arange(16)[::2], SV_dict["control"][ctx], color='k', width=0.4, alpha=0.5, label='conrol')
                ax[i].bar(np.arange(16)[::2] + 0.4, SV_dict["enhanced"][ctx], color='b', width=0.4, alpha=0.5, label='enhanced')
                ax[i].legend(['corrupted', 'control', 'enhanced'], loc=3)
                ax[i].set_xticks(ticks=np.arange(16)[::2], labels=["ctx M", "ctx C", "mR", "mL", "cR", "cL", "OutR", "OutL"])
                for j in [-1, 1, 3, 5, 7, 9, 11, 13]:
                    ax[i].axvline(j, color='k', linestyle='--')
                ax[i].plot([0, 16], [0, 0], color='gray')
                ax[i].set_xlim([-1, 15])
            plt.tight_layout()
            plt.savefig(os.path.join("../", "../", "img", f"{RNN_folder}_epsilon={epsilon}_manipulating_SV.png"))
            # plt.show()
            plt.close(fig)