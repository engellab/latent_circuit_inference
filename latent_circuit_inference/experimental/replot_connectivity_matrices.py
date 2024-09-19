import json

import numpy as np
from matplotlib import pyplot as plt
import os

from latent_circuit_inference.utils.plotting_functions import plot_connectivity

path = "/Users/tolmach/Documents/GitHub/latent_circuit_inference/data/inferred_LCs"
taskname = "CDDM"
score_dict = {}
for af in ["sigmoid", "tanh"]:
    score_dict[af] = {}
    for constrained in [True, False]:
        score_dict[af][f"constrained_{constrained}"] = []
        sfolder = f"CDDM_{af}_constrained={constrained}"
        ssfolders = os.listdir(os.path.join(path, sfolder))
        for ssfolder in ssfolders:
            if ssfolder == ".DS_Store":
                pass
            else:
                sssfolders = os.listdir(os.path.join(path, sfolder, ssfolder))
                sssfolders = [sssfolder for sssfolder in sssfolders if sssfolder != ".DS_Store"]
                for sssfolder in sssfolders:
                    try:
                        LC_path = os.path.join(path, sfolder, ssfolder, sssfolder)
                        LC_params = json.load(open(os.path.join(LC_path, "LC_params.json")))
                        W_inp = np.array(LC_params["W_inp"])
                        W_rec = np.array(LC_params["W_rec"])
                        W_out = np.array(LC_params["W_out"])

                        img_path = os.path.join(LC_path, "connectivity.pdf")
                        plot_connectivity(W_inp=W_inp, W_rec=W_rec, W_out=W_out,
                                          show_inp=False, show_values=False,
                                          show=False,
                                          save=True,
                                          path=img_path)
                    except:
                        pass


