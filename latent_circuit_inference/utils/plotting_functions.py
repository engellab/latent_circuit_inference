import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import matplotlib
# matplotlib.use('Agg')
os.system('python ../../style/style_setup.py')
mm = 1/25.4
from matplotlib import rcParams
# Set global font properties
rcParams['font.family'] = 'helvetica'

def create_optimized_divergent_colormap():
    # Define the colors: soft red, white, soft green, and soft blue
    cdict = {
        'red':   [(0.0, 0.3, 0.3),  # Soft blue at the start
                  (0.5, 1.0, 1.0),  # White in the middle
                  (1.0, 0.8, 0.8)], # Soft red at the end
        'green': [(0.0, 0.4, 0.4),  # Soft blue at the start
                  (0.5, 1.0, 1.0),  # White in the middle
                  (1.0, 0.2, 0.2)], # Soft red at the end
        'blue':  [(0.0, 0.8, 0.8),  # Soft blue at the start
                  (0.5, 1.0, 1.0),  # White in the middle
                  (1.0, 0.3, 0.3)]  # Soft red at the end
    }
    # Create the colormap
    custom_cmap = mcolors.LinearSegmentedColormap('OptimizedMap', segmentdata=cdict, N=256)
    return custom_cmap

# Create the custom colormap
cmap = create_optimized_divergent_colormap()


def plot_connectivity(W_inp, W_rec, W_out, show_inp=False, show=True, show_values=True, save=False, path=None):
    value_thr = 0.05
    num_inputs = W_inp.shape[1]
    num_outputs = W_out.shape[0]
    n_rows = 3 if show_inp else 2

    fig, ax = plt.subplots(n_rows, 1, figsize=(n_rows * 50*mm, 88*mm), constrained_layout=False,
                           gridspec_kw={'height_ratios':
                                            [1, W_rec.shape[0] / num_inputs, num_outputs / num_inputs][-n_rows:]})

    # Create and customize the colorbar
    cbar_ax = fig.add_axes([0.85, 0.12, 0.02, 0.76])
    colormap = create_optimized_divergent_colormap()  # Choose a colormap
    W_inp_min = np.min(W_inp)
    W_inp_max = np.max(W_inp)
    W_rec_min = np.min(W_rec)
    W_rec_max = np.max(W_rec)
    W_out_min = np.min(W_out)
    W_out_max = np.max(W_out)
    if show_inp:
        cap_val = np.max([np.abs(W_inp_min), np.abs(W_inp_max),
                          np.abs(W_rec_min), np.abs(W_rec_max),
                          np.abs(W_out_min), np.abs(W_out_max)])
    else:
        cap_val = np.max([np.abs(W_rec_min), np.abs(W_rec_max),
                          np.abs(W_out_min), np.abs(W_out_max)])
    color_normalizer = mcolors.Normalize(vmin=-cap_val, vmax=cap_val)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=color_normalizer, cmap=colormap), cax=cbar_ax)
    cbar.set_ticks([-cap_val, 0, cap_val])
    cbar.set_ticklabels([-np.round(cap_val, 1), 0, np.round(cap_val, 1)])

    ax_cnt = 0
    if show_inp:
        matrix = W_inp.T
        im_inp = ax[ax_cnt].imshow(matrix, cmap=cmap, vmin=-cap_val, vmax=cap_val, aspect='equal')
        # Add text annotations for each pixel
        if show_values:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.abs(matrix[i, j]) >= value_thr:
                        ax[ax_cnt].text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='black')

        # ax[ax_cnt].set_xticks(np.arange(W_inp.shape[0]))
        # ax[ax_cnt].set_xticklabels(np.arange(1, W_inp.shape[0] + 1))
        ax[ax_cnt].set_xticks([])
        ax[ax_cnt].set_xticklabels([])

        ax[ax_cnt].set_yticks(np.arange(W_inp.shape[1]))
        ax[ax_cnt].set_yticklabels(np.arange(1, W_inp.shape[1] + 1))

        ax_cnt += 1

    matrix = W_rec
    im_rec = ax[ax_cnt].imshow(matrix, cmap=cmap,vmin=-cap_val, vmax=cap_val, aspect='equal')
    if show_values:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.abs(matrix[i, j]) >= value_thr:
                    ax[ax_cnt].text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='black')
    ax[ax_cnt].set_xticks([])
    ax[ax_cnt].set_yticks(np.arange(W_rec.shape[0]))
    ax[ax_cnt].set_yticklabels(np.arange(1, W_rec.shape[0] + 1))
    ax_cnt += 1
    matrix = W_out
    im_out = ax[ax_cnt].imshow(matrix, cmap=cmap, vmin=-cap_val, vmax=cap_val, aspect='equal')
    if show_values:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.abs(matrix[i, j]) >= value_thr:
                    ax[ax_cnt].text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='black')

    ax[-1].set_xticks(np.arange(W_rec.shape[0]))
    ax[-1].set_xticklabels(np.arange(1, W_rec.shape[0] + 1))

    ax[-1].set_yticks(np.arange(W_out.shape[0]))
    ax[-1].set_yticklabels(np.arange(1, W_out.shape[0] + 1))

    if save:
        fig.savefig(path, dpi=300, transparent=True, pad_inches=0.1)
        # path_png = path.split(".pdf")[0] + ".png"
        # plt.savefig(path, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.01)

    if show:
        plt.show()

    plt.close()
    return None


def plot_matrix(mat, vmin=None, vmax=None, show_numbers = False, figsize = (7,7)):
    if vmin is None:
        vmin = np.min(mat)
    if vmax is None:
        vmax = np.max(mat)
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    img = ax.imshow(mat, cmap=cmap, vmin = vmin, vmax = vmax)
    if show_numbers:
        for (i, j), z in np.ndenumerate(mat):
            if np.abs(z) > 0.01:
                ax.text(j, i, str(np.round(z, 2)), ha="center", va="center", color='k', fontsize=7)
    ax.set_xticks(np.arange(mat.shape[1])[::2] + 1)
    ax.set_yticks(np.arange(mat.shape[0])[::2] + 1)
    plt.show()