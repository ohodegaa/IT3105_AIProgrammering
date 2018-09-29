import tfmpl
import numpy as np
import matplotlib.pyplot as plt

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


@tfmpl.figure_tensor
def draw_hinton_plot(matrices, trans=True, max_weight=None, colors=("red", "blue"), headline="Plots", titles=()):
    if type(matrices) != np.array:
        matrices = np.array(matrices)
    if len(titles) == 0:
        titles = ["Plot %d" % i for i in range(len(matrices))]
    fig = tfmpl.create_figure()
    fig.suptitle(headline, fontsize=18)

    axes = fig.subplots(1, len(matrices))

    for matrix, title, ax in zip(matrices, titles, axes):
        if trans:
            matrix = matrix
        if max_weight is None:
            max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(matrix):
            color = colors[0] if w > 0 else colors[1]
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.invert_yaxis()

    return fig
