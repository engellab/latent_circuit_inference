import sys
from copy import deepcopy

sys.path.append("../../")
from rnn_coach.src.PerformanceAnalyzer import *
from latent_circuit_inference.src.utils import *
import numpy as np
from matplotlib import pyplot as plt
from latent_circuit_inference.src.circuit_vizualization import Graph
from matplotlib.patches import Patch
colors, cmap = get_colormaps()
red, blue, bluish, green, orange, lblue, violet = colors

class LCAnalyzer(PerformanceAnalyzerCDDM):
    def __init__(self, rnn_numpy, labels):
        PerformanceAnalyzerCDDM.__init__(self, rnn_numpy)
        self.labels = labels

    def plot_recurrent_matrix(self):
        w_rec = self.RNN.W_rec
        n = self.RNN.N

        fig_w_rec = plt.figure()
        ax = plt.gca()
        im = ax.imshow(self.RNN.W_rec, interpolation='blackman', cmap=cmap)
        fig_w_rec.colorbar(im)
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(self.labels)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(self.labels)
        ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
        plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
                 ha="right", va="center", rotation_mode="anchor")
        plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
                 ha="left", va="center", rotation_mode="anchor")

        for (i, j), z in np.ndenumerate(w_rec):
            if np.abs(z) >= 0.05:
                if z >= -1:
                    ax.text(j, i, str(np.round(z, 2)), ha="center", va="center", color='k')
                if z < -1:
                    ax.text(j, i, str(np.round(z, 2)), ha="center", va="center", color='w')
        # ax.set_title("Connectivity matrix", fontsize = 16, pad=10)
        im = ax.imshow(w_rec, interpolation='none', vmin=-np.max(np.abs(w_rec)), vmax=np.max(np.abs(w_rec)), cmap=cmap)
        fig_w_rec.tight_layout()
        return fig_w_rec

    def plot_recurrent_matrix_comparison(self, w_rec_bar):
        w_rec = self.RNN.W_rec
        n = self.RNN.N

        fig_w_rec_comparison, ax_w_rec_comparison = plt.subplots(1, 2, figsize=(6, 3))
        ax_w_rec_comparison[0].set_xticks(np.arange(n))
        ax_w_rec_comparison[0].set_xticklabels(self.labels)
        ax_w_rec_comparison[0].set_yticks(np.arange(n))
        ax_w_rec_comparison[0].set_yticklabels(self.labels)
        ax_w_rec_comparison[1].set_xticks(np.arange(n))
        ax_w_rec_comparison[1].set_xticklabels(self.labels)
        ax_w_rec_comparison[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        plt.setp([tick.label1 for tick in ax_w_rec_comparison[0].xaxis.get_major_ticks()], rotation=45,
                 ha="right", va="center", rotation_mode="anchor")
        plt.setp([tick.label2 for tick in ax_w_rec_comparison[0].xaxis.get_major_ticks()], rotation=45,
                 ha="left", va="center", rotation_mode="anchor")
        im_w_rec_bar = ax_w_rec_comparison[0].imshow(w_rec_bar, interpolation='none', vmin=-np.max(np.abs(w_rec)),
                                                     vmax=np.max(np.abs(w_rec)), cmap=cmap)
        ax_w_rec_comparison[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        plt.setp([tick.label1 for tick in ax_w_rec_comparison[1].xaxis.get_major_ticks()], rotation=45,
                 ha="right", va="center", rotation_mode="anchor")
        plt.setp([tick.label2 for tick in ax_w_rec_comparison[1].xaxis.get_major_ticks()], rotation=45,
                 ha="left", va="center", rotation_mode="anchor")
        ax_w_rec_comparison[1].set_yticks([])
        im_w_rec = ax_w_rec_comparison[1].imshow(w_rec, interpolation='none', vmin=-np.max(np.abs(w_rec)),
                                                 vmax=np.max(np.abs(w_rec)), cmap=cmap)
        fig_w_rec_comparison.subplots_adjust(right=0.8)
        # cbar_ax = fig_w_rec_comparison.add_axes([1.03, 0.2, 0.02, 0.6])
        # cbar = fig_w_rec_comparison.colorbar(im_w_rec, cax=cbar_ax)
        plt.subplots_adjust(wspace=0.05)
        fig_w_rec_comparison.tight_layout()
        return fig_w_rec_comparison


    def plot_circuit_graph(self):
        w_rec = deepcopy(self.RNN.W_rec)
        positions = [np.array([-0.05, 0.75]),
                     np.array([0.35, 0.7]),
                     np.array([-0.6, 0.4]),
                     np.array([-0.75, 0.0]),
                     np.array([-0.75, -0.4]),
                     np.array([-0.6, -0.8]),
                     np.array([0.6, 0.0]),
                     np.array([0.6, -0.4])]
        np.fill_diagonal(w_rec, 0)
        G = Graph(node_labels=self.labels, positions=positions, W=w_rec)
        G.set_nodes()
        G.set_edges_from_matrix()
        G.optimize_connections()
        G.draw_nodes()
        G.draw_edges()
        plt.tight_layout()
        return G.painter.figure

    def plot_selection_vectors(self, Q, LA_data_lc, LA_data_RNN):
        error_kw = {"ecolor": colors[5], "elinewidth": 1, "capsize": 3, "capthick": 1, "barsabove": True}
        fig, ax = plt.subplots(2, 1, figsize=(4, 8))
        fig.suptitle("Selection vectors", fontsize=16)
        width = 0.4

        legend_elements = [Patch(facecolor=bluish, alpha=0.95, edgecolor='k', label='RNN, Q-projected'),
                           Patch(facecolor=violet, alpha=0.95, edgecolor='k', label='Latent Circuit')]

        for i, ctx in enumerate(["motion", "color"]):
            projected_l_RNN = (np.array(LA_data_RNN[ctx]["l"][:5:-5, :])) @ Q.T
            avg_pr_l_RNN = np.mean(projected_l_RNN, axis=0).flatten()
            avg_pr_l_RNN = avg_pr_l_RNN / np.linalg.norm(avg_pr_l_RNN)
            projected_l_circuit = (np.array(LA_data_lc[ctx]["l"][:5:-5, :]))
            avg_pr_l_circuit = np.mean(projected_l_circuit, axis=0).flatten()
            avg_pr_l_circuit = avg_pr_l_circuit / np.linalg.norm(avg_pr_l_circuit)
            colorlist1 = [bluish for i in np.arange(8)]
            colorlist2 = [violet for i in np.arange(8)]
            rects1 = ax[i].bar(np.arange(len(self.labels)) - width / 2,
                               (avg_pr_l_RNN),
                               tick_label=self.labels,
                               yerr=np.std(projected_l_RNN, axis=0),
                               color=colorlist1, alpha=0.95,
                               ecolor=orange, edgecolor='k', width=width, error_kw=error_kw, label=f"RNN")
            rects2 = ax[i].bar(np.arange(len(self.labels)) + width / 2,
                               (avg_pr_l_circuit),
                               tick_label=self.labels,
                               yerr=np.std(projected_l_circuit, axis=0),
                               color=colorlist2, alpha=0.95,
                               ecolor=orange, edgecolor='k', width=width, error_kw=error_kw, label=f"Latent Circuit")
            ax[i].set_ylim([-0.89, 0.89])
            ax[i].set_xticks(np.arange(len(self.labels)), self.labels, rotation=45)
            ax[i].set_yticks([-0.7, 0.7])
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
            ax[i].axhline(0, color='gray')
            ax[0].legend(handles=legend_elements, loc=1)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.65)
        return fig