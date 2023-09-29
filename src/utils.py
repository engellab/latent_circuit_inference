import sys
import os
from scipy.interpolate import interp1d
sys.path.insert(0, os.getcwd())
sys.path.insert(0, './')
sys.path.insert(0, '../')
import numpy as np
from scipy.sparse import random
from scipy.stats import uniform
from numpy.linalg import eig
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import ListedColormap
import torch
from rnn_coach.src.utils import get_colormaps
colors, cmp = get_colormaps()
red, blue, bluish, green, orange, lblue, violet = colors

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

def get_colormaps():
    # define colors
    color1 = hsv_to_rgb([357.93 / 360, 84.06/100, 81.18/100]) # red
    color2 = hsv_to_rgb([226.73 / 360, 65/100, 64/100]) #blue
    color3 = hsv_to_rgb([246.16 / 360, 33 / 100, 74 / 100]) #bluish
    color4 = hsv_to_rgb([145.14 / 360, 93/100,63/100]) # green
    color5 = hsv_to_rgb([28.32 / 360, 87.35/100, 96.08/100]) # orange
    color6 = hsv_to_rgb([196.11 / 360, 78 / 100, 93 / 100]) #light blue
    color7 = hsv_to_rgb([305 / 360, 53.66 / 100, 64.31 / 100]) # magenta


    colors = [color1, color2, color3, color4, color5, color6, color7]
    white = (1, 1, 1, 1)

    # for i in range(7):
    #     plt.plot(np.arange(9)-1*i, color=eval(f"color{i+1}"), linewidth = 10)
    # plt.show()

    newcolors = np.zeros((256, 4))
    newcolors[:, -1] = 1
    newcolors[:128, 0] = np.linspace(color2[0], white[0], 128)
    newcolors[128:, 0] = np.linspace(white[0], color1[0], 128)
    newcolors[:128, 1] = np.linspace(color2[1], white[1], 128)
    newcolors[128:, 1] = np.linspace(white[1], color1[1], 128)
    newcolors[:128, 2] = np.linspace(color2[2], white[2], 128)
    newcolors[128:, 2] = np.linspace(white[2], color1[2], 128)
    cmp_light = ListedColormap(newcolors)
    return colors, cmp_light


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


def plot_matrix(mat, vmin=None, vmax=None, show_numbers = False, figsize = (7,7)):
    if vmin is None:
        vmin = np.min(mat)
    if vmax is None:
        vmax = np.max(mat)
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    img = ax.imshow(mat, cmap=cmp, vmin = vmin, vmax = vmax)
    if show_numbers:
        for (i, j), z in np.ndenumerate(mat):
            if np.abs(z) > 0.01:
                ax.text(j, i, str(np.round(z, 2)), ha="center", va="center", color='k', fontsize=7)
    ax.set_xticks(np.arange(mat.shape[1])[::2])
    ax.set_yticks(np.arange(mat.shape[0])[::2])
    plt.show()