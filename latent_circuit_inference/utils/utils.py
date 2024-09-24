import json
import sys
import os

from omegaconf import OmegaConf
from scipy.interpolate import interp1d
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import numpy as np
from scipy.sparse import random
from scipy.stats import uniform
from numpy.linalg import eig
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import ListedColormap
import torch

def get_project_root():
    return Path(__file__).parent.parent

def numpify(function_torch):
    return lambda x: function_torch(torch.Tensor(x)).detach().numpy()

def in_the_list(x, x_list, diff_cutoff=1e-6):
    for i in range(len(x_list)):
        diff = np.linalg.norm(x-x_list[i],2)
        if diff < diff_cutoff:
            return True
    return False

def orthonormalize(W):
    for i in range(W.shape[-1]):
        for j in range(i):
            W[:, i] = W[:, i] - W[:, j] * np.dot(W[:, i], W[:, j])
        W[:, i] = W[:, i]/np.linalg.norm(W[:, i])
    return W

def ReLU(x):
    return np.maximum(x, 0)

def generate_recurrent_weights(N, density, sr):
    A = (1.0 / (density * np.sqrt(N))) * np.array(random(N, N, density, data_rvs=uniform(-1, 2).rvs).todense())
    # get eigenvalues
    w, v = eig(A)
    A = A * (sr / np.max(np.abs(w)))
    return A

def sort_eigs(E, R):
    # sort eigenvectors
    data = np.hstack([E.reshape(-1, 1), R.T])
    data = np.array(sorted(data, key=lambda l: np.real(l[0])))[::-1, :]
    E = data[:, 0]
    R = data[:, 1:].T
    return E, R

def make_orientation_consistent(vectors, num_iter=10):
    vectors = np.stack(np.stack(vectors))
    for i in range(num_iter):  # np.min(dot_prod) < 0:
        average_vect = np.mean(vectors, axis=0)
        average_vect /= np.linalg.norm(average_vect)
        dot_prod = vectors @ average_vect
        vectors[np.where(dot_prod < 0)[0], :] *= -1
    return vectors

def cosine_sim(A, B):
    v1 = A.flatten()/np.linalg.norm(A.flatten())
    v2 = B.flatten()/np.linalg.norm(B.flatten())
    return np.round(np.dot(v1, v2),3)

def gini(v, n_points = 1000):
    """Compute Gini coefficient of array of values"""
    v_abs =np.sort(np.abs(v))
    cumsum_v=np.cumsum(v_abs)
    n = len(v_abs)
    vals = np.concatenate([[0], cumsum_v/cumsum_v[-1]])
    dx = 1/n
    x = np.linspace(0, 1, n+1)
    f = interp1d(x=x, y=vals, kind='previous')
    xnew = np.linspace(0, 1, n_points+1)
    dx_new = 1/(n_points)
    vals_new = f(xnew)
    return 1 - 2 * np.trapz(y=vals_new, x=xnew, dx=dx_new)

def sparsity(M, method="gini"):
    a = []
    for i in range(M.shape[0]):
        a.append(eval(f"{method}")(np.abs(M[i, :])))
    return a

def jsonify(dct):
    dct_jsonified = {}
    for key in list(dct.keys()):
        if type(dct[key]) == type({}):
            dct_jsonified[key] = jsonify(dct[key])
        elif type(dct[key]) == np.ndarray:
            dct_jsonified[key] = dct[key].tolist()
        else:
            dct_jsonified[key] = dct[key]
    return dct_jsonified


def mse_scoring(x, y):
    return np.mean((x - y) ** 2)

def R2(x, y):
    return 1.0 - mse_scoring(x, y)/np.var(y)

def get_RNN_conf(path):
    files = os.listdir(path)
    for file in files:
        if "_config" in file:
            if "yaml" in file:
                RNN_conf = OmegaConf.create(OmegaConf.load(os.path.join(path, file)))
            elif "json" in file:
                RNN_conf = OmegaConf.create(json.load(open(os.path.join(path, file), "rb+")))
            break
    return RNN_conf

def get_RNN_data(path):
    files = os.listdir(path)
    for file in files:
        if "_params_" in file:
            rnn_data = json.load(open(os.path.join(path, file), "rb+"))
            break
    return rnn_data