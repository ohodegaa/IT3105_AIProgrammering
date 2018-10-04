import tfmpl
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch  # Needed for dendrograms
import io

import tensorflow as tf


@tfmpl.figure_tensor
def draw_scatter(vals, color):
    fig = tfmpl.create_figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.scatter(*zip(*vals), c=color)
    fig.tight_layout()

    return fig


@tfmpl.figure_tensor
def draw_error_vs_validation(error_vals, validation_vals, colors=("r", "b")):
    fig = tfmpl.create_figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.plot(*zip(*error_vals), c=colors[0])
    ax.plot(*zip(*validation_vals), c=colors[1])

    return fig


def draw_hinton_plot(matrices, trans=True, max_val=None, max_weight=None, scale=True, colors=("red", "blue"),
                     titles=(), step=0):
    headline = "Hinton plot, step %d" % step
    if len(titles) == 0:
        titles = ["Plot %d" % i for i in range(len(matrices))]
    fig = tfmpl.create_figure()
    fig.suptitle(headline, fontsize=18)

    axes = fig.subplots(1, len(matrices))

    for matrix, title, ax in zip(matrices, titles, axes):
        ax.clear()
        if trans:
            matrix = matrix.transpose()
        if max_val is None:  # maxval
            max_val = np.abs(matrix).max()
        if max_weight is None:  # maxsize
            max_weight = 2 ** np.ceil(np.log(max_val) / np.log(2))
        ax.set_title(title)
        ax.patch.set_facecolor('gray')
        ax.set_aspect('auto', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        y_max = (matrix.shape[1] - 1) * max_weight

        for (x, y), w in np.ndenumerate(matrix):
            color = colors[0] if w > 0 else colors[1]
            if scale:
                size = max(0.01, np.sqrt(min(max_weight, max_weight * np.abs(w) / max_val)))
            else:
                size = np.sqrt(min(np.abs(w), max_weight))
            bottom_left = [x - size / 2, (y_max - y) - size / 2]
            rect = plt.Rectangle(bottom_left, size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.invert_yaxis()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    return buf.getvalue()


def draw_dendrogram(features, labels, metric='euclidean', mode='average', title='Dendrogram', orient='top',
                    lrot=90.0):
    fig = tfmpl.create_figure(figsize=(6, 10))
    ax = fig.add_subplot(111)
    print(features)
    cluster_history = sch.linkage(features, method=mode, metric=metric)
    sch.dendrogram(cluster_history, labels=labels, orientation=orient, leaf_rotation=lrot, ax=ax)
    plt.tight_layout()
    ax.set_title(title)
    ax.set_ylabel(metric + ' distance')
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    return buf.getvalue()


"""
class HintonPlotter:

    def __init__(self):
        self.plots = []

    @tfmpl.figure_tensor
    def draw_hinton_plot(self, matrices, trans=True, max_val=None, max_weight=None, scale=True, colors=("red", "blue"),
                         headline="Plots", titles=()):
        if type(matrices) != np.array:
            matrices = np.array(matrices)
        if len(titles) == 0:
            titles = ["Plot %d" % i for i in range(len(matrices))]
        fig = tfmpl.create_figure()
        fig.suptitle(headline, fontsize=18)

        axes = fig.subplots(1, len(matrices))

        for matrix, title, ax in zip(matrices, titles, axes):
            ax.clear()
            if trans:
                matrix = matrix.transpose()
            if max_val is None:  # maxval
                max_val = np.abs(matrix).max()
            if max_weight is None:  # maxsize
                max_weight = 2 ** np.ceil(np.log(max_val) / np.log(2))
            ax.set_title(title)
            ax.patch.set_facecolor('gray')
            ax.set_aspect('auto', 'box')
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

            y_max = (matrix.shape[1] - 1) * max_weight

            for (x, y), w in np.ndenumerate(matrix):
                color = colors[0] if w > 0 else colors[1]
                if scale:
                    size = max(0.01, np.sqrt(min(max_weight, max_weight * np.abs(w) / max_val)))
                else:
                    size = np.sqrt(min(np.abs(w), max_weight))
                bottom_left = [x - size / 2, (y_max - y) - size / 2]
                rect = plt.Rectangle(bottom_left, size, size,
                                     facecolor=color, edgecolor=color)
                ax.add_patch(rect)

            ax.autoscale_view()
            ax.invert_yaxis()

        self.plots.append(fig)
"""
