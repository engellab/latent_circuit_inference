import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)
import os
import sys
import json
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
sys.path.insert(0, "../../../")
import torch
from scipy.stats import zscore
from rnn_coach.src.Task import TaskCDDM
from rnn_coach.src.RNN_torch import RNN_torch
from rnn_coach.src.RNN_numpy import RNN_numpy
from scipy.sparse.linalg import lsqr
from copy import deepcopy
from sklearn.decomposition import PCA
from pathlib import Path

# RNNs = []
# for folder in os.listdir(os.path.join('../', '../', "data", "inferred_LCs")):
#     if (folder == "exemplar_RNN") or (folder == '.DS_Store') or (folder == '0.0077001_20230222-010124'):
#         pass
#     else:
#         if "CDDM;relu" in folder:
#             RNNs.append(folder)
#
# RNNs_filtered = []
# for RNN_folder in RNNs:
#     RNN_score = float(RNN_folder.split("_")[0])
#     RNN_path = os.path.join('../', '../', "rnn_coach", "data", "trained_RNNs", "CDDM", RNN_folder)
#     LC_folder_path = os.path.join('../', '../' "data", "inferred_LCs", RNN_folder)
#     subfolders = os.listdir(LC_folder_path)
#     ind = 0
#     max_score = -10
#     for i, subfolder in enumerate(subfolders):
#         if "8nodes" in subfolder:
#             score = float(subfolder.split("_")[0])
#             if max_score <= score:
#                 max_score = score
#                 ind = i
#     LC_subfolder = subfolders[ind]
#     LC_path = os.path.join('../', "data", "inferred_LCs", RNN_folder, LC_subfolder)
#
#     if max_score > 0.89:
#         RNNs_filtered.append(RNN_folder)
#     # fig = plt.figure(figsize=(10, 5))
#     # img = mpimg.imread(open(os.path.join(LC_path, f"{max_score}_LC_wrec_comparison.png"), 'rb+'))
#     # plt.imshow(img)
#     # if max_score > 0.89:
#     #     plt.savefig(os.path.join("../", "img", f"{RNN_folder}_conncomparison.png"))
#     # plt.show()
#     print(RNNs_filtered)

home = str(Path.home()) + "/Documents/GitHub/"
RNN = "0.0117232_CDDM;relu;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000"
RNNs_path = os.path.join(home, "latent_circuit_inference", "data", "inferred_LCs")
num_points = 31
offset = 5
RNN_score = float(RNN.split("_")[0])
RNN_path = os.path.join(RNNs_path, RNN)
LCs = os.listdir(RNN_path)
try:
    LCs.remove('.DS_Store')
except:
    pass
LC_scores = [float(LC.split("_")[0]) for LC in LCs]
LC_scores_pr = [float(LC.split("_")[1]) for LC in LCs]
LC_scores,LC_scores_pr, LCs = zip(*sorted(zip(LC_scores,LC_scores_pr, LCs), reverse=True))
top_LC = LCs[0]
ind = 0
LC_path = os.path.join(home, "latent_circuit_inference", "data", "inferred_LCs", RNN, top_LC)
print(LC_path)
LC_data = json.load(open(os.path.join(LC_path, f"{LC_scores[0]}_{LC_scores_pr[0]}_LC_params.json"), "rb+"))

RNN_data = json.load(open(os.path.join(RNN_path, f"{LC_scores[0]}_{LC_scores_pr[0]}_LC_8nodes;encoding", f"{RNN_score}_params_CDDM.json"), "rb+"))
RNN_config_file = json.load(open(os.path.join(RNN_path, f"{LC_scores[0]}_{LC_scores_pr[0]}_LC_8nodes;encoding", f"{RNN_score}_config.json"), "rb+"))

U = np.array(LC_data["U"])
q = np.array(LC_data["q"])
Q = U.T @ q

w_out = np.array(LC_data["W_out"])
w_rec = np.array(LC_data["W_rec"])
w_inp = np.array(LC_data["W_inp"])
n = LC_data["N"]
N = LC_data["N"]
dt = LC_data["dt"]
tau = LC_data["tau"]

W_out = np.array(RNN_data["W_out"])
W_rec = np.array(RNN_data["W_rec"])
W_inp = np.array(RNN_data["W_inp"])
bias_rec = np.array(RNN_data["bias_rec"])
y_init = np.array(RNN_data["y_init"])
activation = RNN_config_file["activation"]
mask = np.array(RNN_config_file["mask"])
input_size = RNN_config_file["num_inputs"]
output_size = RNN_config_file["num_outputs"]
task_params = RNN_config_file["task_params"]
n_steps = task_params["n_steps"]
sigma_inp = RNN_config_file["sigma_inp"]
sigma_rec = RNN_config_file["sigma_rec"]

task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
seed = np.random.randint(1000000)
print(f"seed: {seed}")
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
rng = torch.Generator(device=torch.device(device))
if not seed is None:
    rng.manual_seed(seed)

if activation == 'relu':
    activation_RNN = lambda x: torch.maximum(x, torch.tensor(0))
elif activation == 'tanh':
    activation_RNN = torch.tanh
elif activation == 'sigmoid':
    activation_RNN = lambda x: 1 / (1 + torch.exp(-x))
elif activation == 'softplus':
    activation_RNN = lambda x: torch.log(1 + torch.exp(5 * x))

RNN = RNN_torch(N=W_rec.shape[0], dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                activation=activation_RNN, random_generator=rng, device=device,
                sigma_rec=sigma_rec, sigma_inp=sigma_inp)
RNN_params = {"W_inp": W_inp,
              "W_rec": W_rec,
              "W_out": W_out,
              "b_rec": None,
              "y_init": np.zeros(W_rec.shape[0])}
RNN.set_params(RNN_params)

input_batch, target_batch, conditions_batch = task.get_batch()
n_trials = len(conditions_batch)
RNN.sigma_rec = RNN.sigma_inp = torch.tensor(0, device=RNN.device)
y, predicted_output_rnn = RNN(torch.from_numpy(input_batch.astype("float32")))
Y = np.hstack([y.detach().numpy()[:, :, i] for i in range(y.shape[-1])])

q_variables = np.einsum("ij,jkl->ikl", Q.T, y.detach().numpy())
# TDR
Z = zscore(Y, axis=1)
z = Z.reshape(-1, n_trials, n_steps)
z = np.swapaxes(z, 1, 2)
# PCA on Z
pca = PCA(n_components=12)
pca.fit(Z.T)
PCs = pca.components_
D = PCs.T @ PCs
Z_pca = D @ Z
z_pca = Z_pca.reshape(-1, n_trials, n_steps)
z_pca = np.swapaxes(z_pca, 1, 2)
context = np.array(
    [(1 if conditions_batch[i]['context'] == 'motion' else -1) for i in range(len(conditions_batch))])
motion_coh = np.array([conditions_batch[i]['motion_coh'] for i in range(len(conditions_batch))])
color_coh = np.array([conditions_batch[i]['color_coh'] for i in range(len(conditions_batch))])
choice = np.array([conditions_batch[i]['correct_choice'] for i in range(len(conditions_batch))])
F = np.hstack([context.reshape(-1, 1),
               motion_coh.reshape(-1, 1),
               color_coh.reshape(-1, 1),
               choice.reshape(-1, 1),
               np.ones((n_trials, 1))])
B = np.zeros((Z.shape[0], n_steps, F.shape[1]))
for i in range(Z.shape[0]):
    for t in range(n_steps):
        betas_i_t = lsqr(F, z_pca[i, t, :], damp=0)[0]
        B[i, t, :] = deepcopy(betas_i_t)

ind_cont = np.argmax(np.linalg.norm(B[:, :, 0], axis=0))
ind_motion = np.argmax(np.linalg.norm(B[:, :, 1], axis=0))
ind_color = np.argmax(np.linalg.norm(B[:, :, 2], axis=0))
ind_choice = np.argmax(np.linalg.norm(B[:, :, 3], axis=0))
context_direction = B[:, ind_cont, 0]
motion_direction = B[:, ind_motion, 1]
color_direction = B[:, ind_color, 2]
choice_direction = B[:, ind_choice, 3]
B_max = np.hstack([context_direction.reshape(-1, 1),
                   motion_direction.reshape(-1, 1),
                   color_direction.reshape(-1, 1),
                   choice_direction.reshape(-1, 1)])
B_orth, R = np.linalg.qr(B_max)

Vars = (B_orth.T[:, :] @ Z_pca[:, :]).reshape(-1, n_trials, n_steps)
Vars = np.swapaxes(Vars, 1, 2)

fig, ax = plt.subplots(2, 2, figsize=(10, 7))
plt.suptitle("Motion information")
ax[0, 0].plot(q_variables[3, :, :225] - q_variables[2, :, :225], color='r', alpha=0.1)
ax[0, 0].plot(q_variables[3, :, 225:] - q_variables[2, :, 225:], color='b', alpha=0.1)
ax[0, 1].plot(Vars[1, :, :225], color='r', alpha=0.1, label = 'relevant')
ax[0, 1].plot(Vars[1, :, 225:], color='b', alpha=0.1, label = 'irrelevant')
ax[0, 1].set_ylim([-12, 12])

ax[1, 0].plot(q_variables[5, :, :225] - q_variables[4, :, :225], color='b', alpha=0.1, label = 'irrelevant')
ax[1, 0].plot(q_variables[5, :, 225:] - q_variables[4, :, 225:], color='r', alpha=0.1, label = 'relevant')
ax[1, 1].plot(Vars[2, :, :225], color='b', alpha=0.1)
ax[1, 1].plot(Vars[2, :, 225:], color='r', alpha=0.1)
ax[1, 1].set_ylim([-12, 12])
plt.legend()
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
ax[0, 0].title.set_text('qmR - qmL, motion context')
ax[0, 1].title.set_text('TDR motion, motion context')
ax[1, 0].title.set_text('qcR - qcL, color context')
ax[1, 1].title.set_text('TDR color, color context')
# plt.savefig(os.path.join("../", "../", "img", f"{RNN_folder}_QvsTDR.png"))
plt.close(fig)
plt.show()

