import os
import sys
import pickle
import json
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
sys.path.insert(0, "../../../")
from matplotlib.patches import Patch
from rnn_coach.src.utils import get_colormaps
colors, cmp = get_colormaps()
red, blue, bluish, green, orange, lblue, violet = colors
mm = 1/25.4  # inches in mm
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)
from rnn_coach.src.Task import TaskCDDM, TaskCDDMplus
from rnn_coach.src.RNN_numpy import RNN_numpy
from rnn_coach.src.DynamicSystemAnalyzer import DynamicSystemAnalyzerCDDM
from copy import deepcopy
import pandas as pd
from pathlib import Path
import pickle

def plot_matrix(mat):
    names = ["ctx m", "ctx c", "mr", "ml", "cr", "cl", "OutR", "OutL"]
    n = 8
    fig, ax = plt.subplots(1, figsize = (6, 3))

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(names)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(names)

    # Set ticks on both sides of axes on
    # ax_w_rec_comparison[0].tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    # Rotate and align bottom ticklabels
    plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
             ha="right", va="center", rotation_mode="anchor")
    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
             ha="left", va="center",rotation_mode="anchor")
    im_w_rec = ax.imshow(mat, interpolation='none', vmin=-np.max(np.abs(w_rec)), vmax = np.max(np.abs(w_rec)), cmap=cmp)


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.68, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(im_w_rec, cax=cbar_ax)
    # cbar.ax.set_ylabel('value', rotation=270, labelpad=10, fontsize=16)
    plt.subplots_adjust(wspace = 0.05)
    return fig

home = str(Path.home())
# if os.uname().nodename == 'della-gpu.princeton.edu':
#     host = 'DELLA'
# else:
#     host = 'local'
#     home += "/Documents/GitHub/"

task_name = "CDDM"
RNNs_path = os.path.join(home, "rnn_coach", "data", "trained_RNNs", task_name)
RNNs = []
for folder in os.listdir(RNNs_path):
    if (folder == '.DS_Store'):
        pass
    else:
        if "relu" in folder:
            RNNs.append(folder)

names = []
scores = []
Ns = []
lmbdos = []
lmbdrs = []
lrs = []
activations = []
tags = []
maxiters = []
# on DELLA
# if host == 'DELLA':
top_RNNs = ["0.0077129_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0078828_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0078976_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0079038_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0079381_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0079561_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0079603_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0079618_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0079909_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0080135_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0080566_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0080809_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0080838_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0080912_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0080926_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0081178_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0081334_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0081378_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.008143_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0081447_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0081486_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0081499_CDDM;relu;N=93;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0081577_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0081671_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0081731_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082014_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082115_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082116_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082137_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082141_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082217_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082225_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082255_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082259_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.008233_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082376_CDDM;relu;N=91;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082376_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082485_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082524_CDDM;relu;N=92;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082712_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082755_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082843_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0082988_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.008306_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083069_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083083_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083139_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083198_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083226_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083266_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083387_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083511_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083565_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083576_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083585_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083631_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083724_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083769_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083809_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.008384_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083881_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0083955_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0084013_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0084014_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0084085_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0084142_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.008415_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0084232_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0084414_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0084591_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0084667_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0084715_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0084802_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.008482_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0084993_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0085186_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0085341_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0085344_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0085503_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0085516_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0085544_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0086042_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0086053_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0086129_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0086175_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0086197_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0086401_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.008641_CDDM;relu;N=92;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0086443_CDDM;relu;N=93;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0086473_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0086827_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0086976_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0087091_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.008714_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0087219_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0088062_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0088584_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0088914_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0088925_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "0.0090541_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000"]
# if host == 'local':
#     pass
#     top_RNNs = ["0.0077175_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0078732_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0079036_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0079341_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0079439_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0079477_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0079531_CDDM;relu;N=100;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0079612_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0079655_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0079709_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0079841_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0079912_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0079937_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080035_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080107_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080133_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080209_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080274_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.008031_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080315_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080402_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080607_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080613_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080626_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080673_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080706_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080832_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0080962_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0081005_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0081093_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.008121_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0081459_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0081468_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0081489_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0081498_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0081544_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0081708_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0081766_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0081927_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0081985_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082023_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082056_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082088_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082207_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082221_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082233_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082253_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082388_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.00824_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082408_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082439_CDDM;relu;N=93;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082593_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082723_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082739_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082742_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082804_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082868_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.008287_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082922_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0082947_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0083185_CDDM;relu;N=100;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0083242_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0083442_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0083566_CDDM;relu;N=92;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0083622_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.008376_CDDM;relu;N=91;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.008394_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.008414_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0084217_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0084264_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.00843_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0084331_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0084336_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0084602_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0084705_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0084769_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.008481_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0084897_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0084953_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085015_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085037_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085303_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085317_CDDM;relu;N=100;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085333_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085369_CDDM;relu;N=99;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085396_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.008555_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085577_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085745_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085797_CDDM;relu;N=95;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085945_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085967_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0085977_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0086433_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0086799_CDDM;relu;N=94;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0087479_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0087543_CDDM;relu;N=98;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0088071_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0088574_CDDM;relu;N=97;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0088651_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0089895_CDDM;relu;N=96;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000", "0.0167548_CDDM;relu;N=48;lmbdo=0.3;lmbdr=0.3;lr=0.002;maxiter=3000"]
# calculating means
W_recs_pr = []
w_recs = []
SV_motion = []
sv_motion = []
SV_color = []
sv_color = []
JAC_motion = []
jac_motion = []
JAC_color = []
jac_color = []
num_points = 31
offset = 5
for num_rnn in range(len(top_RNNs)):
    print(num_rnn)
    try:
        RNN_subfolder = top_RNNs[num_rnn]
        RNN_score = float(top_RNNs[num_rnn].split("_")[0])
        RNN_path = os.path.join(RNNs_path, RNN_subfolder)
        RNN_data = json.load(open(os.path.join(RNN_path, f"{RNN_score}_params_{task_name}.json"), "rb+"))
        RNN_config_file = json.load(open(os.path.join(RNN_path, f"{RNN_score}_config.json"), "rb+"))
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
        dt = RNN_config_file["dt"]
        tau = RNN_config_file["tau"]

        LC_folder = RNN_subfolder
        LC_folder_path = os.path.join(home, "latent_circuit_inference", "data", "inferred_LCs",
                                      LC_folder)

        subfolders = os.listdir(LC_folder_path)
        varianses = []
        variances_pr = []
        names = []
        for i, subfolder in enumerate(subfolders):
            if "8nodes" in subfolder or "8-nodes" in subfolder:
                score = float(subfolder.split("_")[0])
                score_pr = float(subfolder.split("_")[1])
                varianses.append(score)
                variances_pr.append(score_pr)
                names.append(subfolder)
        lc_df = pd.DataFrame({"name": names, "variance": varianses, "variance_pr": variances_pr})
        top_LCs = lc_df.sort_values("variance", ascending=False)["name"].tolist()
        LC_subfolder = top_LCs[0]
        score = float(LC_subfolder.split("_")[0])
        score_pr = float(LC_subfolder.split("_")[1])
        LC_path = os.path.join(LC_folder_path, LC_subfolder)
        LC_data = json.load(open(os.path.join(LC_path, f"{score}_{score_pr}_LC_params.json"), "rb+"))
        U = np.array(LC_data["U"])
        q = np.array(LC_data["q"])
        Q = U.T @ q
        w_out = np.array(LC_data["W_out"])
        w_rec = np.array(LC_data["W_rec"])
        w_inp = np.array(LC_data["W_inp"])
        N = LC_data["N"]
        dt = LC_data["dt"]
        tau = LC_data["tau"]

        # loading up the task
        task = TaskCDDM(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
        input_batch, target_batch, conditions_batch = task.get_batch()
        n_trials = len(conditions_batch)

        # creating instances of lc and RNN
        lc = RNN_numpy(N=8, dt=dt, tau=tau, W_inp=w_inp, W_rec=w_rec, W_out=w_out)
        RNN = RNN_numpy(N=W_rec.shape[0], dt=1, tau=tau, W_inp=W_inp, W_rec=W_rec, W_out=W_out)

        dsa_RNN = DynamicSystemAnalyzerCDDM(RNN)
        params = {"fun_tol": 0.05,
                  "diff_cutoff": 1e-4,
                  "sigma_init_guess": 10,
                  "patience": 10,
                  "stop_length": 10,
                  "mode": "approx"}

        dsa_lc = DynamicSystemAnalyzerCDDM(lc)
        params = {"fun_tol": 0.05,
                  "diff_cutoff": 1e-4,
                  "sigma_init_guess": 10,
                  "patience": 10,
                  "stop_length": 10,
                  "mode": "approx"}

        #instead of calculating, load it
        LA_data_RNN = pickle.load(open(os.path.join(RNNs_path, RNN_subfolder, f"{RNN_score}_LA_data.pkl"), "rb+"))
        # dsa_RNN.calc_LineAttractor_analytics(N_points=101)
        dsa_lc.calc_LineAttractor_analytics(N_points=num_points)
        LA_data_lc = dsa_lc.LA_data

        # selection vectors:
        projected_l_RNN_motion = (np.array(LA_data_RNN["motion"]["l"][:-offset])) @ Q
        avg_pr_l_RNN_motion = np.mean(projected_l_RNN_motion, axis=0).flatten()
        avg_pr_l_RNN_motion = avg_pr_l_RNN_motion / np.linalg.norm(avg_pr_l_RNN_motion)
        projected_l_circuit_motion = (np.array(LA_data_lc["motion"]["l"][offset:-offset]))
        avg_pr_l_circuit_motion = np.mean(projected_l_circuit_motion, axis=0).flatten()
        avg_pr_l_circuit_motion = avg_pr_l_circuit_motion / np.linalg.norm(avg_pr_l_circuit_motion)

        projected_l_RNN_color = (np.array(LA_data_RNN["color"]["l"][offset:-offset])) @ Q
        avg_pr_l_RNN_color = np.mean(projected_l_RNN_color, axis=0).flatten()
        avg_pr_l_RNN_color = avg_pr_l_RNN_color / np.linalg.norm(avg_pr_l_RNN_color)
        projected_l_circuit_color = (np.array(LA_data_lc["color"]["l"][offset:-offset]))
        avg_pr_l_circuit_color = np.mean(projected_l_circuit_color, axis=0).flatten()
        avg_pr_l_circuit_color = avg_pr_l_circuit_color / np.linalg.norm(avg_pr_l_circuit_color)

        # Jacobians
        Js_motion = LA_data_RNN["motion"]["jac"][offset:-offset]
        js_motion = LA_data_lc["motion"]["jac"][offset:-offset]
        Js_color = LA_data_RNN["color"]["jac"][offset:-offset]
        js_color = LA_data_lc["color"]["jac"][offset:-offset]

        projected_Ws_motion_d = [Q.T @ (J + np.eye(J.shape[0])) @ Q for J in Js_motion]
        ws_motion_d = [j + np.eye(8) for j in js_motion]
        projected_W_motion = np.mean(np.array(projected_Ws_motion_d), axis=0)
        #     projected_W_motion = np.array(projected_Ws_motion_d)[len(Js_motion)//2,:, :]
        w_motion = np.mean(np.array(ws_motion_d), axis=0)
        #     w_motion = np.array(ws_motion_d)[len(Js_motion)//2,:, :]

        projected_Ws_color_d = [Q.T @ (J + np.eye(J.shape[0])) @ Q for J in Js_color]
        ws_color_d = [j + np.eye(8) for j in js_color]
        projected_W_color = np.mean(np.array(projected_Ws_color_d), axis=0)
        #     projected_W_color = np.array(projected_Ws_color_d)[len(Js_motion)//2,:, :]
        w_color = np.mean(np.array(ws_color_d), axis=0)
        #     w_color = np.array(ws_color_d)[len(Js_motion)//2,:, :]

        W_recs_pr.append(Q.T @ W_rec @ Q)
        w_recs.append(deepcopy(w_rec))
        SV_motion.append(deepcopy(avg_pr_l_RNN_motion))
        sv_motion.append(deepcopy(avg_pr_l_circuit_motion))
        SV_color.append(deepcopy(avg_pr_l_RNN_color))
        sv_color.append(deepcopy(avg_pr_l_circuit_color))

        JAC_motion.append(deepcopy(projected_W_motion))
        jac_motion.append(deepcopy(w_motion))
        JAC_color.append(deepcopy(projected_W_color))
        jac_color.append(deepcopy(w_color))

    except FileNotFoundError:
        print(f"No latent circuits for: {top_RNNs[num_rnn]}")
        pass

    # save the data
    data = {"W_recs_pr" : W_recs_pr,
           "w_recs" : w_recs,
           "SV_motion" : SV_motion,
           "SV_color" : SV_color,
           "sv_motion" : sv_motion,
           "sv_color" : sv_color,
           "JAC_motion" : JAC_motion,
           "JAC_color" : JAC_color,
           "jac_motion" : jac_motion,
           "jac_color" : jac_color}



pickle.dump(data, open(os.path.join(home, "latent_circuit_inference", "data", "data_across_RNNs.pkl"), "wb+"))

data = pickle.load(open(os.path.join(home, "latent_circuit_inference", "data", "data_across_RNNs.pkl"), "rb+"))
W_recs_pr = data["W_recs_pr"]
w_recs = data["w_recs"]
SV_motion = data["SV_motion"]
SV_color = data["SV_color"]
sv_motion = data["sv_motion"]
sv_color = data["sv_color"]
JAC_motion = data["JAC_motion"]
JAC_color = data["JAC_color"]
jac_motion = data["jac_motion"]
jac_color = data["jac_color"]

w_rec_mean = np.median(np.array(w_recs), axis = 0)
fig_w_rec_mean = plot_matrix(w_rec_mean)
plt.savefig(os.path.join(home, "latent_circuit_inference", "img", "w_rec_mean.pdf"), dpi=300, bbox_inches = 'tight', transparent=True)

W_rec_mean = np.median(np.array(W_recs_pr), axis = 0)
fig_W_rec_mean = plot_matrix(W_rec_mean)
plt.savefig(os.path.join(home, "latent_circuit_inference", "img", "W_rec_mean.pdf"), dpi=300, bbox_inches = 'tight', transparent=True)

jac_motion_avg = np.median(np.array(jac_motion), axis = 0)
fig_jac_motion_avg = plot_matrix(jac_motion_avg)
plt.savefig(os.path.join(home, "latent_circuit_inference", "img", "jac_motion_avg.pdf"), dpi=300, bbox_inches = 'tight', transparent=True)

JAC_motion_avg = np.median(np.array(JAC_motion), axis = 0)
fig_JAC_motion_avg = plot_matrix(JAC_motion_avg)
plt.savefig(os.path.join(home, "latent_circuit_inference", "img", "JAC_motion_avg.pdf"), dpi=300, bbox_inches = 'tight', transparent=True)

jac_color_avg = np.median(np.array(jac_color), axis = 0)
fig_jac_color_avg = plot_matrix(jac_color_avg)
plt.savefig(os.path.join(home, "latent_circuit_inference", "img", "jac_color_avg.pdf"), dpi=300, bbox_inches = 'tight', transparent=True)

JAC_color_avg = np.median(np.array(JAC_color), axis = 0)
fig_JAC_color_avg = plot_matrix(JAC_color_avg)
plt.savefig(os.path.join(home, "latent_circuit_inference", "img", "JAC_color_avg.pdf"), dpi=300, bbox_inches = 'tight', transparent=True)

SV_motion_avg = np.mean(np.array(SV_motion), axis = 0)
SV_color_avg = np.mean(np.array(SV_color), axis = 0)
sv_motion_avg = np.mean(np.array(sv_motion), axis = 0)
sv_color_avg = np.mean(np.array(sv_color), axis = 0)
SV = {"motion" : SV_motion_avg, "color" : SV_color_avg}
sv = {"motion" : sv_motion_avg, "color" : sv_color_avg}

var_names = ["ctx m", "ctx c", "mR", "mL", "cR", "cL", "OutR", "OutL"]

for i, ctx in enumerate(["motion", "color"]):
    avg_pr_l_list = []
    error_kw = {"ecolor": 'k', "elinewidth": 1, "capsize": 3, "capthick": 1, "barsabove": True}
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    width = 0.4

    legend_elements = [Patch(facecolor=violet, alpha=0.95, edgecolor='k', label='RNN, Q-projected'),
                       Patch(facecolor=bluish, alpha=0.95, edgecolor='k', label='Latent Circuit')]
    avg_pr_l_RNN = SV[ctx]
    avg_pr_l_circuit = sv[ctx]
    colorlist1 = [violet for i in np.arange(8)]
    colorlist2 = [bluish for i in np.arange(8)]
    rects1 = ax.bar(np.arange(8) - width / 2,
                    (avg_pr_l_RNN),
                    tick_label=var_names,
                    #               yerr = np.std(projected_l_RNN, axis = 0),
                    color=colorlist1, alpha=0.95,
                    ecolor=violet, edgecolor='k', width=width, error_kw=error_kw, label=f"RNN")
    rects2 = ax.bar(np.arange(8) + width / 2,
                    (avg_pr_l_circuit),
                    tick_label=var_names,
                    #               yerr = np.std(projected_l_circuit, axis = 0),
                    color=colorlist2, alpha=0.95,
                    ecolor=violet, edgecolor='k', width=width, error_kw=error_kw, label=f"Latent Circuit")
    ax.set_ylim([-0.89, 0.89])
    ax.set_xticks(np.arange(8), ["ctx m", "ctx c", "mR", "mL", "cR", "cL", "OutR", "OutL"], rotation=90, fontsize=16)
    ax.set_yticks([-0.7, 0.7], [-0.7, 0.7], rotation=90, fontsize=16)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('right')

    ax.axhline(0, color='gray')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.65)
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.2, 1.4), loc=2, frameon=False)  # , rotation=90)
    plt.savefig(os.path.join(home, "latent_circuit_inference", "img", f"selection_vectors_comparison_vertical_{ctx}.pdf"), dpi=300, bbox_inches = 'tight', transparent=True)
