import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.colors import LinearSegmentedColormap

os.system('python ../../style/style_setup.py')
mm = 1/25.4

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
    custom_cmap = LinearSegmentedColormap('OptimizedMap', segmentdata=cdict, N=256)
    return custom_cmap

# Create the custom colormap
cmap = create_optimized_divergent_colormap()



def plot_connectivity(W_inp, W_rec, W_out, show_inp=False, show=True, show_values=True, save=False, path=None):
    value_thr = 0.05
    num_inputs = W_inp.shape[1]
    num_outputs = W_out.shape[0]
    n_rows = 3 if show_inp else 2

    fig, ax = plt.subplots(n_rows, 1, figsize=(n_rows * 40*mm, 88*mm), constrained_layout=False,
                           gridspec_kw={'height_ratios':
                                            [1, W_rec.shape[0] / num_inputs, num_outputs / num_inputs][-n_rows:]})
    ax_cnt = 0
    if show_inp:
        matrix = W_inp.T
        ax[ax_cnt].imshow(matrix, cmap=cmap, vmin=-0.5, vmax=0.5, aspect='equal')
        # Add text annotations for each pixel
        if show_values:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.abs(matrix[i, j]) >= value_thr:
                        ax[ax_cnt].text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='black')
        ax[ax_cnt].set_xticks([])
        ax[ax_cnt].set_yticks(np.arange(W_inp.shape[1]))
        ax_cnt += 1

    matrix = W_rec
    ax[ax_cnt].imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect='equal')
    if show_values:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.abs(matrix[i, j]) >= value_thr:
                    ax[ax_cnt].text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='black')
    ax[ax_cnt].set_xticks([])
    ax[ax_cnt].set_yticks(np.arange(W_rec.shape[0]))
    ax_cnt += 1
    matrix = W_out
    ax[ax_cnt].imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect='equal')
    if show_values:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.abs(matrix[i, j]) >= value_thr:
                    ax[ax_cnt].text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='black')
    ax[-1].set_xticks(np.arange(W_rec.shape[0]))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.tight_layout()
    if save:
        plt.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        plt.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')
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
    ax.set_xticks(np.arange(mat.shape[1])[::2])
    ax.set_yticks(np.arange(mat.shape[0])[::2])
    plt.show()