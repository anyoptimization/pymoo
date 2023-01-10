import matplotlib.pyplot as plt


def plot_pairs_3d(first, second, colors=("indigo", "firebrick"), **kwargs):
    
    fig, ax = plt.subplots(1, 2, subplot_kw={'projection':'3d'}, **kwargs)

    ax[0].scatter(
        *first[1].T,
        color=colors[0], label=first[0], marker="o",
    )
    ax[0].set_ylabel("$f_2$")
    ax[0].set_xlabel("$f_1$")
    ax[0].set_zlabel("$f_3$")
    ax[0].legend()

    ax[1].scatter(
        *second[1].T,
        color=colors[1], label=second[0], marker="o",
    )
    ax[1].set_ylabel("$f_2$")
    ax[1].set_xlabel("$f_1$")
    ax[1].set_zlabel("$f_3$")
    ax[1].legend()

    ax[0].view_init(elev=30, azim=30)
    ax[1].view_init(elev=30, azim=30)

    fig.tight_layout()
    plt.show()


def plot_pairs_2d(first, second, colors=("indigo", "firebrick"), **kwargs):
    
    fig, ax = plt.subplots(1, 2, **kwargs)

    ax[0].scatter(
        *first[1].T,
        color=colors[0], label=first[0], marker="o",
    )
    ax[0].set_ylabel("$f_2$")
    ax[0].set_xlabel("$f_1$")
    ax[0].legend()

    ax[1].scatter(
        *second[1].T,
        color=colors[1], label=second[0], marker="o",
    )
    ax[1].set_ylabel("$f_2$")
    ax[1].set_xlabel("$f_1$")
    ax[1].legend()

    fig.tight_layout()
    plt.show()