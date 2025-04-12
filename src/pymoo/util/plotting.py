import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def plot(*args, show=True, labels=None, no_fill=False, **kwargs):
    F = args[0]

    if F.ndim == 1:
        print("Cannot plot a one dimensional array.")
        return

    n_dim = F.shape[1]

    if n_dim == 2:
        ret = plot_2d(*args, labels=labels, no_fill=no_fill, **kwargs)
    elif n_dim == 3:
        ret = plot_3d(*args, labels=labels, no_fill=no_fill, **kwargs)
    else:
        print("Cannot plot a %s dimensional array." % n_dim)
        return

    if labels:
        plt.legend()

    if show:
        plt.show()

    return ret


def plot_3d(*args, no_fill=False, labels=None, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, F in enumerate(args):

        if no_fill:
            kwargs["s"] = 20
            kwargs["marker"] = '.'
            kwargs["facecolors"] = (0, 0, 0, 0)
            kwargs["edgecolors"] = 'r'

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

    print("Saving: ", path_to_file)


def plot_problem_surface(problem, n_samples, plot_type="wireframe", cmap="summer", show=True, return_figure=False):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except:
        raise Exception("Please install 'matplotlib' to use the plotting functionality.")

    fig = plt.figure()

    if problem.n_var == 1 and problem.n_obj == 1:

        X = np.linspace(problem.xl[0], problem.xu[0], num=n_samples)[:, None]
        Y = problem.evaluate(X, return_values_of=["F"])
        ax = plt.plot(X, Y)

    elif problem.n_var == 2 and problem.n_obj == 1:

        X_range = np.linspace(problem.xl[0], problem.xu[0], num=n_samples)
        Y_range = np.linspace(problem.xl[1], problem.xu[1], num=n_samples)
        X, Y = np.meshgrid(X_range, Y_range)

        A = np.zeros((n_samples * n_samples, 2))
        counter = 0
        for i, x in enumerate(X_range):
            for j, y in enumerate(Y_range):
                A[counter, 0] = x
                A[counter, 1] = y
                counter += 1

        F = np.reshape(problem.evaluate(A, return_values_of=["F"]), (n_samples, n_samples))

        # Plot the surface.
        if plot_type == "wireframe":
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(X, Y, F)
        elif plot_type == "contour":
            CS = plt.contour(X, Y, F)
            plt.clabel(CS, inline=1, fontsize=10)
        elif plot_type == "wireframe+contour":
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, F, cmap=cmap, rstride=1, cstride=1)
            ax.contour(X, Y, F, 10, linestyles="solid", offset=-1)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$f(x)$")
            ax.view_init(45, 45)
        else:
            raise Exception("Unknown plotting method.")


    else:
        raise Exception("Can only plot single with less than two variables and one objective.")

    if show:
        plt.tight_layout()
        plt.show()

    if return_figure:
        return fig, ax
