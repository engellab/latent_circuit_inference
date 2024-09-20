import sys

from latent_circuit_inference.utils.plotting_functions import plot_connectivity
from latent_circuit_inference.utils.utils import set_paths
sys.path.append("../experimental/")
sys.path.append("../")
sys.path.append("../../")
from pathlib import Path
import numpy as np
import json
import os
import torch
import trainRNNbrain
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask
from trainRNNbrain.tnns.RNN_torch import *
from trainRNNbrain.rnns.RNN_numpy import *
from trainRNNbrain.tasks import *
from trainRNNbrain.datasaver.DataSaver import *
from latent_circuit_inference.LatentCircuit import LatentCircuit
from latent_circuit_inference.LatentCircuitFitter import *
from latent_circuit_inference.LCAnalyzer import *
from matplotlib import pyplot as plt
from omegaconf import OmegaConf, DictConfig
import hydra


OmegaConf.register_new_resolver("eval", eval)
os.environ['HYDRA_FULL_ERROR'] = '1'
show = False
# @hydra.main(version_base="1.3", config_path="../../configs/", config_name=f"base")
@hydra.main(version_base="1.3", config_path="../../configs/", config_name=f"base")
def run_LCI(cfg: DictConfig) -> None:
    RNN_parent_folder = cfg.RNN_parent_folder
    RNN_subfolder = cfg.RNN_subfolder
    taskname = RNN_parent_folder.split("_")[0]
    activation_name = RNN_parent_folder.split("_")[1]
    constrained = bool(RNN_parent_folder.split("_")[2].split("=")[1])
    print(cfg.n_units)

    tag = f"{activation_name}_constrained={constrained}"
    trained_RNNs_path, data_save_path = set_paths(parent_folder=RNN_parent_folder)
    RNN_folder_full_path = os.path.join(trained_RNNs_path, RNN_subfolder)

    #loading other necessary configs:
    LC_model_conf = OmegaConf.load(os.path.join(cfg.configs_path,
                                                f"LC_model/lc_{activation_name}.yaml"))
    task_specific_conf = OmegaConf.load(os.path.join(cfg.configs_path,
                                                     f"task_specific_constraints/{taskname}_{cfg.n_units}units.yaml"))

    seed = np.random.randint(1000000)
    print(f"seed: {seed}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    rng = torch.Generator(device=torch.device(device))
    if not seed is None:
        rng.manual_seed(seed)

    # defining RNN
    RNN_conf = get_RNN_conf(RNN_folder_full_path)
    rnn_data = get_RNN_data(RNN_folder_full_path)

    # Convert to a dictionary-like structure to allow dynamic key assignments
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    cfg["LC_model"] = LC_model_conf
    cfg["task_specific_constraints"] = task_specific_conf
    cfg["RNN_config"] = RNN_conf

    # defining the task
    # a little crutch here:
    cfg.RNN_config.task._target_ = f"rnn_coach.Tasks.Task{taskname}.Task{taskname}"
    task_conf = prepare_task_arguments(cfg_task=cfg.RNN_config.task, dt=cfg.LC_model.dt)
    task = hydra.utils.instantiate(task_conf)
    cfg["task"] = task_conf

    activation_name = rnn_data["activation_name"]
    activation_slope = rnn_data.get("activation_slope", 1.0 if activation_name != "sigmoid" else 7.5)

    for i in range(cfg.n_repeats):
        rnn_torch = RNN_torch(N=rnn_data["N"],
                              dt=rnn_data["dt"], tau=rnn_data["tau"],
                              exc_to_inh_ratio=cfg.RNN_config.model.exc_to_inh_ratio,
                              input_size=np.array(rnn_data["W_inp"]).shape[1],
                              output_size=np.array(rnn_data["W_out"]).shape[0],
                              activation_name=activation_name,
                              activation_slope=activation_slope,
                              seed=seed,
                              sigma_rec=cfg.RNN_config.model.sigma_rec,
                              sigma_inp=cfg.RNN_config.model.sigma_inp)
        RNN_params = {"W_inp": np.array(rnn_data["W_inp"]),
                      "W_rec": np.array(rnn_data["W_rec"]),
                      "W_out": np.array(rnn_data["W_out"]),
                      "bias_rec": None if rnn_data["bias_rec"] is None else np.array(rnn_data["bias_rec"]),
                      "y_init": np.zeros(rnn_data["N"]),
                      "activation_name": activation_name,
                      "activation_slope": activation_slope}
        rnn_torch.set_params(RNN_params)

        # defining LC
        lc = hydra.utils.instantiate(OmegaConf.merge(LC_model_conf, task_specific_conf))
        lc.random_generator = rng

        # defining LC_fitter
        criterion = torch.nn.MSELoss()
        fitter = LatentCircuitFitter(LatentCircuit=lc,
                                     RNN=rnn_torch,
                                     Task=task,
                                     N_PCs=cfg.LC_fitter.N_PCs,
                                     encoding=cfg.LC_fitter.encoding,
                                     max_iter=cfg.LC_fitter.max_iter,
                                     tol=cfg.LC_fitter.tol,
                                     lr=cfg.LC_fitter.lr,
                                     criterion=criterion,
                                     lambda_w=cfg.LC_fitter.lambda_w,
                                     Qinitialization=cfg.LC_fitter.Qinitialization)
        lc_inferred, train_losses, val_losses, lc_params = fitter.run_training()

        # defining circuit
        circuit_building_params = {key: lc_params[key] for key in lc_params.keys() if key not in ["q", "U"]}
        circuit = RNN_numpy(**circuit_building_params)
        circuit.y = np.zeros(cfg.task_specific_constraints.N)

        # defining numpy RNN
        RNN_building_params = RNN_params
        for key in ["N", "dt", "tau"]:
            RNN_building_params[key] = rnn_data[key]
        N_RNN = rnn_data["N"]
        RNN = RNN_numpy(**RNN_params)
        RNN.y = np.zeros(N_RNN)

        U = lc_params["U"]
        q = lc_params["q"]
        Q = U.T @ q
        w_rec_bar = Q.T @ RNN_params["W_rec"] @ Q
        node_labels = np.arange(lc_params["W_rec"].shape[0])
        analyzer = LCAnalyzer(circuit, labels=node_labels)
        input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
        mask = get_training_mask(cfg.RNN_config.task, dt=lc_params["dt"])

        #MSE mse_score_RNN
        score_function = lambda x, y: np.mean((x - y) ** 2)
        mse_score = analyzer.get_validation_score(scoring_function=score_function,
                                                  input_batch=input_batch_valid,
                                                  target_batch=target_batch_valid,
                                                  mask=mask,
                                                  sigma_rec=cfg.RNN_config.model.sigma_rec,
                                                  sigma_inp=cfg.RNN_config.model.sigma_inp)
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
        print(scores)

        data_save_folder = os.path.join(data_save_path, cfg.RNN_subfolder, f"{r2_tot}_{r2_proj}_LC_{tag}")
        datasaver = DataSaver(data_save_folder)
        datasaver.save_data(scores, f"{r2_tot}_{r2_proj}_LC_scores.json")

        datasaver.save_data(cfg, f"full_LC_config.yaml")
        datasaver.save_data(jsonify(lc_params), f"LC_params.json")
        datasaver.save_data(jsonify(RNN_params), f"RNN_params.json")

        fig_w_rec = analyzer.plot_recurrent_matrix()
        datasaver.save_figure(fig_w_rec, f"{r2_tot}_{r2_proj}_LC_wrec.png")
        if show: plt.show()

        fig_w_out = analyzer.plot_output_matrix()
        datasaver.save_figure(fig_w_out, f"{r2_tot}_{r2_proj}_LC_wout.png")
        if show: plt.show()

        path = os.path.join(data_save_folder, "connectivity.pdf")
        plot_connectivity(circuit.W_inp, circuit.W_rec, circuit.W_out,
                          show_inp=False, show_values=False,
                          show=show,
                          save=True,
                          path=path)


        print(f"Plotting random trials")
        inds = np.random.choice(np.arange(input_batch_valid.shape[-1]), 10)
        inputs = input_batch_valid[..., inds]
        targets = target_batch_valid[..., inds]
        conditions = [conditions_valid[ind] for ind in inds]
        fig_trials = analyzer.plot_trials(inputs, targets, mask,
                                          conditions=conditions,
                                          sigma_rec=RNN_conf.model.sigma_rec, sigma_inp=RNN_conf.model.sigma_inp)
        datasaver.save_figure(fig_trials, f"{r2_tot}_{r2_proj}_LC_random_trials.png")
        if show: plt.show()

if __name__ == "__main__":
    run_LCI()
