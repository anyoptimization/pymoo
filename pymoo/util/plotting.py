import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def plot(*args, show=True, labels=None, no_fill=False,**kwargs):
    F = args[0]

    if F.ndim == 1:
        print("Cannot plot a one dimensional array.")
        return

    n_dim = F.shape[1]

    if n_dim == 2:
        ret = plot_2d(*args, labels=labels, no_fill=no_fill, **kwargs)
    elif n_dim == 3:
        ret = plot_3d(*args, labels=labels, **kwargs)
    else:
        print("Cannot plot a %s dimensional array." % n_dim)
        return

    if labels:
        plt.legend()

    if show:
        plt.show()

    return ret


def plot_3d(*args, labels=None, **kwargs):
    fig = plt.figure()
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')

    for i, F in enumerate(args):
        if labels:
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], label=labels[i], **kwargs)
        else:
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], **kwargs)

    return ax



def plot_2d(*args, labels=None, no_fill=False):

    if no_fill:
        kwargs = dict(
            s=20,
            facecolors='none',
            edgecolors='r'
        )
    else:
        kwargs = {}

    for i, F in enumerate(args):
        if labels:
            plt.scatter(F[:, 0], F[:, 1], label=labels[i], **kwargs)
        else:
            plt.scatter(F[:, 0], F[:, 1], **kwargs)


def animate(path_to_file, H, problem=None, func_iter=None, plot_min=None, plot_max=None):
    if H.ndim != 3 or H.shape[2] != 2:
        print("Can only animate a two dimensional set of arrays.")
        return

    fig = plt.figure()
    ax = plt.gca()

    # plot the pareto front if it is known for the problem
    if problem is not None:
        pf = problem.pareto_front()
        plt.scatter(pf[:, 0], pf[:, 1], label='Pareto Front', s=60, facecolors='none', edgecolors='r')

    # plot the initial population
    _F = H[0, :, :]
    scat = plt.scatter(_F[:, 0], _F[:, 1])
    plt.title("0")

    if func_iter is not None:
        func_iter(ax, H[0])

    # the update method
    def update(n):
        _F = H[n, :, :]
        scat.set_offsets(_F)

        # get the bounds for plotting and add padding
        min = np.min(_F, axis=0) - 0.1
        max = np.max(_F, axis=0) + 0.1

        # set the scatter object with padding
        ax.set_xlim(min[0], max[0])
        ax.set_ylim(min[1], max[1])

        if func_iter is not None:
            func_iter(ax, H[n])

        plt.title(n)

    # create the animation
    ani = animation.FuncAnimation(fig, update, frames=H.shape[0])

    # write the file
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=6, bitrate=1800)
    ani.save(path_to_file, writer=writer)

    print("Saving: ",  path_to_file)
