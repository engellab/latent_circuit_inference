import numpy as np
from matplotlib import pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from scipy.optimize import minimize
from scipy.stats import rv_continuous


def Rotate(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def bezier_to_point_mindist(p0, p1, p2, q):

    def bezier_func(t, p0, p1, p2):
        a = p0 - 2 * p1 + p2
        b = p1 - p2
        c = p2
        return a * t ** 2 + 2 * b * t + c

    def fun(t, p0, p1, p2, q):
        return np.linalg.norm(bezier_func(t, p0, p1, p2) - q)

    res = minimize(fun, args=(p0, p1, p2, q), x0=np.array([0]), method="SLSQP",
                   constraints={"type": "ineq", "fun": lambda x: 0})
    dist = res.fun
    closest_point = bezier_func(res.x, p0, p1, p2)
    return closest_point, dist

class CustomDistribution(rv_continuous):
    def _rvs(self,  size=None, random_state=None):
        return random_state.standard_normal(size)


class Painter():
    def __init__(self, figsize=(7, 7)):
        fig_circuit, ax = plt.subplots(1, 1, figsize=figsize)
        self.figure = fig_circuit
        self.ax = ax
        self.node_patches = []
        self.arrow_patches = []
        self.ax.set_ylim([-1.1, 1.1])
        self.ax.set_xlim([-1.1, 1.1])
        self.ax.set_aspect('equal')
        self.ax.axis('off')

    def Bezier_arrow(self, posA, posB, capstyle, bezier_point=None, color='r', alpha=1.0, lw=2, theta=np.pi / 8,
                     head_shaft_ratio=0.05):

        Path = mpath.Path
        if bezier_point is None:
            bezier_point = (np.array(posB) + np.array(posA)) / 2

        path_data = [(Path.MOVETO, posA),
                     (Path.CURVE3, bezier_point),
                     (Path.LINETO, posB)]
        codes, verts = zip(*path_data)
        path_shaft = mpath.Path(verts, codes)
        patch_shaft = mpatches.PathPatch(path_shaft, fill=False, alpha=alpha, edgecolor=color, lw=lw, capstyle='round')

        central_shaft = -(np.array(posB) - np.array(bezier_point))
        central_shaft /= np.linalg.norm(central_shaft)
        path_data = []
        if capstyle == 'arrow':
            init_point = np.array(posB) + 0.5 * head_shaft_ratio * central_shaft
            left_arrow_point = head_shaft_ratio * (central_shaft @ Rotate(theta)) + np.array(posB)
            right_arrow_point = head_shaft_ratio * (central_shaft @ Rotate(-theta)) + np.array(posB)
            path_data.extend([(Path.MOVETO, init_point),
                              (Path.LINETO, left_arrow_point),
                              (Path.LINETO, posB),
                              (Path.LINETO, right_arrow_point),
                              (Path.CLOSEPOLY, (0, 0))])
            codes, verts = zip(*path_data)
            path_arrow = mpath.Path(verts, codes)
            patch_arrow = mpatches.PathPatch(path_arrow, fill=True, alpha=alpha/2, edgecolor=color, facecolor=color,
                                             lw=lw)
        if capstyle == 'circle':
            r = 0.02;
            N = 24
            point = (np.array(posB) - r * central_shaft)
            path_data.append((Path.MOVETO, point))
            for n in range(N):
                point = np.array(posB) - r * central_shaft @ Rotate((n / N) * 2 * np.pi)
                path_data.append((Path.LINETO, point))
            path_data.append((Path.CLOSEPOLY, point))
            codes, verts = zip(*path_data)
            path_arrow = mpath.Path(verts, codes)
            patch_arrow = mpatches.PathPatch(path_arrow, fill=True, alpha=alpha, edgecolor='k', facecolor=color, lw=1)
        return patch_shaft, patch_arrow


class Graph():
    def __init__(self, W_inp, W_rec, W_out,
                 labels=None,
                 positions=None,
                 R = 0.8,
                 cutoff_weight=0.05,
                 r = 0.1,
                 default_pos_clr='r',
                 default_neg_clr='b'):
        self.W_inp = W_inp
        self.W_rec = W_rec
        self.W_out = W_out
        self.N = W_rec.shape[0]
        self.R = R
        self.r = r
        if not (labels is None):
            self.labels = labels
        else:
            self.labels = np.arange(self.N).tolist()
        self.rec_edge_dict = {}
        self.inp_edge_dict = {}
        self.out_edge_dict = {}
        self.node_dict = {}
        if (positions is None):
            self.positions = self.get_circular_layout(R=R)
        else:
            self.positions = positions
        self.cutoff_weight = cutoff_weight
        self.default_pos_clr = default_pos_clr
        self.default_neg_clr = default_neg_clr


    def get_circular_layout(self, R):
        positions = []
        for i in range(self.N):
            positions.append(R * np.array([-np.cos(2 * np.pi * i/self.N), -np.sin(2 * np.pi * i/self.N)]))
        return positions


    def set_nodes(self, fill=True, facecolor='skyblue', edgecolor='k', show_label=True):
        for l, label in enumerate(self.labels):
            self.node_dict[label] = {}
            self.node_dict[label]["radius"] = self.r
            self.node_dict[label]["fill"] = fill
            self.node_dict[label]["facecolor"] = facecolor
            self.node_dict[label]["edgecolor"] = edgecolor
            self.node_dict[label]["show_label"] = show_label
            self.node_dict[label]["pos"] = self.positions[l]
        return None

    def set_rec_edge(self, strength, node_from, node_to, bezier_point,
                     color=None,
                     cap='default',
                     ls='-', alpha=1.0):

        j = self.labels.index(node_from)
        i = self.labels.index(node_to)
        self.rec_edge_dict[(j, i)] = {}
        self.rec_edge_dict[(j, i)]["node_from"] = node_from
        self.rec_edge_dict[(j, i)]["node_to"] = node_to
        self.rec_edge_dict[(j, i)]["strength"] = strength
        if color is None:
            color = self.default_pos_clr if (strength >= 0) else self.default_neg_clr
            self.rec_edge_dict[(j, i)]["color"] = color
        if cap == 'default':
            self.rec_edge_dict[(j, i)]["capstyle"] = 'arrow' if (strength >= 0) else 'circle'
        self.rec_edge_dict[(j, i)]["lw"] = strength
        self.rec_edge_dict[(j, i)]["alpha"] = alpha
        self.rec_edge_dict[(j, i)]["ls"] = ls
        self.rec_edge_dict[(j, i)]["bezier_point"] = bezier_point
        return None

    def set_inp_edge(self, strength, inp_from, node_to, bezier_point,
                     color=None,
                     cap='default',
                     ls='-', alpha=1.0):
        j = inp_from
        i = self.labels.index(node_to)
        self.inp_edge_dict[(j, i)] = {}
        self.inp_edge_dict[(j, i)]["input_from"] = inp_from
        self.inp_edge_dict[(j, i)]["node_to"] = node_to
        self.inp_edge_dict[(j, i)]["strength"] = strength
        if color is None:
            color = self.default_pos_clr if (strength >= 0) else self.default_neg_clr
            self.inp_edge_dict[(j, i)]["color"] = color
        if cap == 'default':
            self.inp_edge_dict[(j, i)]["capstyle"] = 'arrow' if (strength >= 0) else 'circle'
        self.inp_edge_dict[(j, i)]["lw"] = strength
        self.inp_edge_dict[(j, i)]["alpha"] = alpha
        self.inp_edge_dict[(j, i)]["ls"] = ls
        self.inp_edge_dict[(j, i)]["bezier_point"] = bezier_point
        return None

    def set_rec_edges_from_matrix(self):
        max_W = np.max(np.abs(self.W_rec))
        for i in range(self.W_rec.shape[0]):
            for j in range(self.W_rec.shape[1]):
                if np.abs(self.W_rec[i, j]) >= self.cutoff_weight:
                    alpha = np.abs(self.W_rec[i, j]) / max_W
                    node_from = self.labels[j]
                    node_to = self.labels[i]
                    posA = self.node_dict[node_from]["pos"]
                    posB = self.node_dict[node_to]["pos"]
                    bezier_point = (np.array(posB) + np.array(posA)) / 2
                    self.set_rec_edge(strength=np.sign(self.W_rec[i, j]) * 1 * (self.W_rec[i, j]) ** 2,
                                      bezier_point=bezier_point,
                                      node_from=self.labels[j],
                                      node_to=self.labels[i],
                                      alpha=alpha)
        return None

    def set_inp_edges_from_matrix(self):
        max_W = np.max(np.abs(self.W_inp))
        for i in range(self.W_inp.shape[0]):
            for j in range(self.W_inp.shape[1]):
                if np.abs(self.W_inp[i, j]) >= self.cutoff_weight:
                    alpha = np.abs(self.W_inp[i, j]) / max_W
                    node_to = self.labels[i]

                    posA = 1.1 * self.node_dict[node_to]["pos"]
                    posB = self.node_dict[node_to]["pos"]
                    bezier_point = (np.array(posB) + np.array(posA)) / 2
                    self.set_inp_edge(strength=3 * np.sign(self.W_inp[i, j]) * (self.W_inp[i, j]) ** 2,
                                      bezier_point=bezier_point,
                                      inp_from=j,
                                      node_to=self.labels[i],
                                      alpha=alpha)
        return None

    def draw_nodes(self):
        for l, label in enumerate(self.labels):
            radius = self.node_dict[label]["radius"]
            fill = self.node_dict[label]["fill"]
            facecolor = self.node_dict[label]["facecolor"]
            edgecolor = self.node_dict[label]["edgecolor"]
            patch = mpatches.Circle(self.positions[l],
                                    radius=radius, fill=fill,
                                    facecolor=facecolor,
                                    edgecolor=edgecolor,
                                    label=label)
            lbl = self.painter.ax.annotate(label, xy=self.positions[l], fontsize=9,
                                           verticalalignment="center",
                                           horizontalalignment="center")
            self.painter.ax.add_patch(patch)
        return None

    def draw_edges(self):
        for edge in list(self.rec_edge_dict.keys()):
            node_from = self.labels[edge[0]]
            node_to = self.labels[edge[1]]
            pos_node_from = self.node_dict[node_from]["pos"]
            pos_node_to = self.node_dict[node_to]["pos"]
            bezier_point = self.rec_edge_dict[edge]["bezier_point"]
            color = self.rec_edge_dict[edge]["color"]

            rad = self.node_dict[node_from]["radius"]
            posA = pos_node_from + rad * (bezier_point - pos_node_from) / np.linalg.norm((bezier_point - pos_node_from))
            posB = pos_node_to + rad * (bezier_point - pos_node_to) / np.linalg.norm((bezier_point - pos_node_to))

            patch_shaft, patch_arrow = self.painter.Bezier_arrow(posA=posA, posB=posB, bezier_point=bezier_point,
                                                                 color=color, lw=self.rec_edge_dict[edge]["lw"],
                                                                 capstyle=self.rec_edge_dict[edge]["capstyle"],
                                                                 alpha=self.rec_edge_dict[edge]["alpha"])
            self.painter.ax.add_patch(patch_shaft)
            self.painter.ax.add_patch(patch_arrow)

        for edge in list(self.inp_edge_dict.keys()):
            node_to = self.labels[edge[1]]
            pos_node_to = self.node_dict[node_to]["pos"]
            bezier_point = self.inp_edge_dict[edge]["bezier_point"]
            color = self.inp_edge_dict[edge]["color"]

            rad = self.node_dict[node_to]["radius"]
            posA =  1.15 * self.node_dict[node_to]["pos"] + rad * self.node_dict[node_to]["pos"]/np.linalg.norm(self.node_dict[node_to]["pos"])
            bezier_point = posA
            posB = pos_node_to + rad * (bezier_point - pos_node_to) / np.linalg.norm((bezier_point - pos_node_to))

            patch_shaft, patch_arrow = self.painter.Bezier_arrow(posA=posA, posB=posB, bezier_point=bezier_point,
                                                                 color=color, lw=self.inp_edge_dict[edge]["lw"],
                                                                 capstyle=self.inp_edge_dict[edge]["capstyle"],
                                                                 alpha=self.inp_edge_dict[edge]["alpha"])
            self.painter.ax.add_patch(patch_shaft)
            self.painter.ax.add_patch(patch_arrow)
        return None


    # def make_animation(self, fig, ax):
    #     return None

    def curve_connections(self):
        for e, edge in enumerate(self.rec_edge_dict.keys()):
            node_from = self.labels.index(self.rec_edge_dict[edge]["node_from"])
            node_to = self.labels.index(self.rec_edge_dict[edge]["node_to"])
            midpoint = (self.positions[node_to] + self.positions[node_from])/2
            vector = (midpoint - self.positions[node_from])
            offset_vector = vector @ Rotate(np.pi/2) / np.linalg.norm(vector)
            self.rec_edge_dict[edge]["bezier_point"] = midpoint + self.R * (1 - np.cos(np.pi / self.N)) * offset_vector

    # def optimize_connections(self):
    #     bezier_midpoints = np.array([self.rec_edge_dict[edge]["bezier_point"] for edge in self.rec_edge_dict.keys()])
    #     bezier_midpoints += 0.1 * np.random.randn(*bezier_midpoints.shape)
    #     positions_from = np.array(
    #         [self.node_dict[self.rec_edge_dict[edge]["node_from"]]["pos"] for edge in self.rec_edge_dict.keys()])
    #     positions_to = np.array(
    #         [self.node_dict[self.rec_edge_dict[edge]["node_to"]]["pos"] for edge in self.rec_edge_dict.keys()])
    #     true_midpoints = (positions_to + positions_from) / 2
    #     node_positions = np.array(self.positions)
    #
    #     def gaussian(point1, point2, sigma=0.15):
    #         C = (1.0 / np.sqrt(2 * np.pi * sigma ** 2))
    #         return C * np.sum(np.exp(-(point1 - point2) ** 2 / (2 * sigma ** 2)))
    #
    #     def objective(x, rec_edge_dict, node_dict):
    #         n_points = true_midpoints.shape[0]
    #         n_nodes = len(node_dict.keys())
    #         # node_positions = np.array([node["pos"] for node in node_dict.keys()])
    #         bezier_midpoints = x.reshape(*true_midpoints.shape)
    #
    #         term_1 = 0
    #         term_2 = 0
    #         for e, edge in enumerate(rec_edge_dict.keys()):
    #             node_from = self.labels[edge[0]]
    #             node_to = self.labels[edge[1]]
    #             midpoint = (node_dict[node_to]["pos"] + node_dict[node_from]["pos"]) / 2
    #             dist = (node_dict[node_to]["pos"] - node_dict[node_from]["pos"])
    #             direction = dist / np.linalg.norm(dist)
    #             right_dir = direction @ Rotate(np.pi / 2)
    #             shift = bezier_midpoints[e, :] - midpoint
    #             term_1 += 1 * (np.dot(right_dir, shift) - 0.1) ** 2
    #             term_2 += 3 * np.dot(direction, shift) ** 2
    #         return term_1 + term_2
    #
    #     x0 = bezier_midpoints.flatten() + np.random.randn()
    #     res = minimize(objective, x0=x0, args=(self.rec_edge_dict, self.node_dict), method="SLSQP")
    #     new_bezier_points = res.x.reshape(*true_midpoints.shape)
    #     for e, edge in enumerate(self.rec_edge_dict.keys()):
    #         self.rec_edge_dict[edge]["bezier_point"] = new_bezier_points[e, :]
    #     return None


if __name__ == '__main__':
    import numpy as np
    from scipy.sparse import random
    from scipy import stats
    from numpy.random import default_rng
    from matplotlib import pyplot as plt

    rng = default_rng()
    X = CustomDistribution(seed=rng)
    Y = X()  # get a frozen version of the distribution
    N = 10
    W_rec = 0.8 * random(N, N, density=0.4, random_state=rng, data_rvs=Y.rvs).toarray()
    np.fill_diagonal(W_rec, 0)
    W_inp = np.zeros((N, 3))
    W_inp[:3, :3] = np.eye(3)
    W_out = None
    G = Graph(W_inp=W_inp, W_rec=W_rec, W_out=W_out, R=0.8, r=0.1, cutoff_weight=0.1)
    G.set_nodes()
    G.set_rec_edges_from_matrix()
    G.set_inp_edges_from_matrix()
    G.curve_connections()
    G.painter = Painter()
    G.draw_nodes()
    G.draw_edges()
    plt.tight_layout()
    plt.show()


