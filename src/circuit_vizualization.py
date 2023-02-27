import numpy as np
from matplotlib import pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from scipy.optimize import minimize


def Rotate(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

def bezier_to_point_mindist(p0, p1, p2, q):

    def bezier_func(t, p0, p1, p2):
        a = p0 - 2 * p1 + p2
        b = p1 - p2
        c = p2
        return a * t ** 2 + 2 * b * t + c

    def fun(t, p0, p1, p2, q):
        return np.linalg.norm(bezier_func(t, p0, p1, p2) - q)**2
    res = minimize(fun, args=(p0, p1, p2, q), x0=np.array([0]), method="SLSQP", constraints={"type": "ineq", "fun": lambda x: 0})
    return res.fun


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from scipy.optimize import minimize


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


class Painter():
    def __init__(self):
        fig, ax = plt.subplots()
        self.figure = fig
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
            patch_arrow = mpatches.PathPatch(path_arrow, fill=True, alpha=alpha, edgecolor=color, facecolor=color,
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
    def __init__(self, node_labels, positions, W, cutoff_weight=0.05,
                 default_pos_clr='r', default_neg_clr='b'):
        self.W = W
        self.labels = node_labels
        self.N = len(self.labels)
        self.edge_dict = {}
        self.node_dict = {}
        self.positions = positions
        self.cutoff_weight = cutoff_weight
        self.default_pos_clr = default_pos_clr
        self.default_neg_clr = default_neg_clr
        self.painter = Painter()

    def set_nodes(self, radius=0.1, fill=True, facecolor='skyblue', edgecolor='k', show_label=True):
        for l, label in enumerate(self.labels):
            self.node_dict[label] = {}
            self.node_dict[label]["radius"] = radius
            self.node_dict[label]["fill"] = fill
            self.node_dict[label]["facecolor"] = facecolor
            self.node_dict[label]["edgecolor"] = edgecolor
            self.node_dict[label]["show_label"] = show_label
            self.node_dict[label]["pos"] = self.positions[l]
        return None

    def set_edge(self, strength, node_from, node_to, bezier_point,
                 color=None,
                 cap='default',
                 ls='-', alpha=1.0):

        j = self.labels.index(node_from)
        i = self.labels.index(node_to)
        self.edge_dict[(j, i)] = {}
        self.edge_dict[(j, i)]["node_from"] = node_from
        self.edge_dict[(j, i)]["node_to"] = node_to
        self.edge_dict[(j, i)]["strength"] = strength
        if color is None:
            color = self.default_pos_clr if (strength >= 0) else self.default_neg_clr
            self.edge_dict[(j, i)]["color"] = color
        if cap == 'default':
            self.edge_dict[(j, i)]["capstyle"] = 'arrow' if (strength >= 0) else 'circle'
        self.edge_dict[(j, i)]["lw"] = strength
        self.edge_dict[(j, i)]["alpha"] = alpha
        self.edge_dict[(j, i)]["ls"] = ls
        self.edge_dict[(j, i)]["bezier_point"] = bezier_point
        return None

    def set_edges_from_matrix(self):
        max_W = np.max(np.abs(self.W))
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                if np.abs(self.W[i, j]) >= self.cutoff_weight:
                    alpha = np.abs(self.W[i, j]) / max_W
                    node_from = self.labels[j]
                    node_to = self.labels[i]
                    posA = self.node_dict[node_from]["pos"]
                    posB = self.node_dict[node_to]["pos"]
                    bezier_point = (np.array(posB) + np.array(posA)) / 2
                    self.set_edge(strength=5 * (self.W[i, j]),  # np.sign(self.W[i, j]) * 3 * (self.W[i, j])**2
                                  bezier_point=bezier_point,
                                  node_from=self.labels[j],
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
        for edge in list(self.edge_dict.keys()):
            node_from = self.labels[edge[0]]
            node_to = self.labels[edge[1]]
            pos_node_from = self.node_dict[node_from]["pos"]
            pos_node_to = self.node_dict[node_to]["pos"]
            bezier_point = self.edge_dict[edge]["bezier_point"]
            color = self.edge_dict[edge]["color"]

            rad = self.node_dict[node_from]["radius"]
            posA = pos_node_from + rad * (bezier_point - pos_node_from) / np.linalg.norm((bezier_point - pos_node_from))
            posB = pos_node_to + rad * (bezier_point - pos_node_to) / np.linalg.norm((bezier_point - pos_node_to))

            patch_shaft, patch_arrow = self.painter.Bezier_arrow(posA=posA, posB=posB, bezier_point=bezier_point,
                                                                 color=color, lw=self.edge_dict[edge]["lw"],
                                                                 capstyle=self.edge_dict[edge]["capstyle"],
                                                                 alpha=self.edge_dict[edge]["alpha"])
            self.painter.ax.add_patch(patch_shaft)
            self.painter.ax.add_patch(patch_arrow)
        return None

    def optimize_connections(self):
        bezier_midpoints = np.array([self.edge_dict[edge]["bezier_point"] for edge in self.edge_dict.keys()])
        bezier_midpoints += 0.1 * np.random.randn(*bezier_midpoints.shape)
        positions_from = np.array(
            [self.node_dict[self.edge_dict[edge]["node_from"]]["pos"] for edge in self.edge_dict.keys()])
        positions_to = np.array(
            [self.node_dict[self.edge_dict[edge]["node_to"]]["pos"] for edge in self.edge_dict.keys()])
        true_midpoints = (positions_to + positions_from) / 2
        node_positions = np.array(self.positions)

        def gaussian(point1, point2, sigma=0.15):
            C = (1.0 / np.sqrt(2 * np.pi * sigma ** 2))
            return C * np.sum(np.exp(-(point1 - point2) ** 2 / (2 * sigma ** 2)))

        def objective(x, edge_dict, node_dict):
            n_points = true_midpoints.shape[0]
            n_nodes = len(node_dict.keys())
            # node_positions = np.array([node["pos"] for node in node_dict.keys()])
            bezier_midpoints = x.reshape(*true_midpoints.shape)

            term_1 = 0
            term_2 = 0
            for e, edge in enumerate(edge_dict.keys()):
                node_from = self.labels[edge[0]]
                node_to = self.labels[edge[1]]
                midpoint = (node_dict[node_to]["pos"] + node_dict[node_from]["pos"]) / 2
                dist = (node_dict[node_to]["pos"] - node_dict[node_from]["pos"])
                direction = dist / np.linalg.norm(dist)
                right_dir = direction @ Rotate(np.pi / 2)
                shift = bezier_midpoints[e, :] - midpoint
                term_1 += 1 * (np.dot(right_dir, shift) - 0.1) ** 2
                term_2 += 3 * np.dot(direction, shift) ** 2
            return term_1 + term_2 + 0.1 * np.sum(bezier_midpoints**2)

        x0 = bezier_midpoints.flatten() + np.random.randn()
        res = minimize(objective, x0=x0, args=(self.edge_dict, self.node_dict), method="SLSQP")
        new_bezier_points = res.x.reshape(*true_midpoints.shape)
        for e, edge in enumerate(self.edge_dict.keys()):
            self.edge_dict[edge]["bezier_point"] = new_bezier_points[e, :]
        return None