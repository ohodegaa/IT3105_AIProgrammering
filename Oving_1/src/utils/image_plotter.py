import tfmpl


@tfmpl.figure_tensor
def draw_scatter(vals, color):
    figs = tfmpl.create_figures(1, figsize=(4, 4))
    for i, fig in enumerate(figs):
        ax = fig.add_subplot(111)
        ax.scatter(*zip(*vals), c=color)
        fig.tight_layout()

    return figs


@tfmpl.figure_tensor
def draw_error_vs_validation(error_vals, validation_vals, colors=("r", "b")):
    figs = tfmpl.create_figures(1, figsize=(4, 4))
    for i, fig in enumerate(figs):
        ax = fig.add_subplot(111)
        ax.plot(*zip(*error_vals), c=colors[0])
        ax.plot(*zip(*validation_vals), c=colors[1])

    return figs
