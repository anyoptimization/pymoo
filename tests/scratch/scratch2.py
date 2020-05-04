from pymoo.factory import get_reference_directions

ref_dirs = get_reference_directions("das-dennis", 10, n_partitions=15)

print(len(ref_dirs))