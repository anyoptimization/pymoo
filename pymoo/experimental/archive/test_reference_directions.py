import numpy as np

from pymoo.util.reference_directions import get_ref_dirs_from_section


def uniform_reference_directions(n_partitions, n_dim):
    ref_dirs = []
    ref_dir = np.full(n_dim, np.inf)
    __uniform_reference_directions(ref_dirs, ref_dir, n_partitions, n_dim, n_partitions, 0)
    return np.concatenate(ref_dirs, axis=0)


def __uniform_reference_directions(ref_dirs, ref_dir, n_partitions, n_dim, beta, depth):
    if depth == n_dim - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            __uniform_reference_directions(ref_dirs, np.copy(ref_dir), n_dim, n_partitions, beta - i, depth + 1)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    ref_array = uniform_reference_directions(12, 3)
    ref_array = get_ref_dirs_from_section(3, 12)

    print(np.sum(ref_array, axis=1))

    fig = plt.figure()
    from mpl_toolkits.mplot3d import Axes3D

    ax = fig.add_subplot(111, projection='3d')

    x = ref_array[:, 0]
    y = ref_array[:, 1]
    z = ref_array[:, 2]

    ax.scatter(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
