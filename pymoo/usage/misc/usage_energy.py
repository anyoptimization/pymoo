from pymoo.factory import get_reference_directions
from pymoo.util.ref_dirs.construction import ConstructionBasedReferenceDirectionFactory
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory
from pymoo.util.ref_dirs.energy_layer import LayerwiseRieszEnergyReferenceDirectionFactory
from pymoo.util.ref_dirs.reduction import ReductionBasedReferenceDirectionFactory
from pymoo.util.ref_dirs.sample_and_map import RandomSamplingAndMap
from pymoo.visualization.scatter import Scatter

n_dim = 3
n_points = 100

fac = LayerwiseRieszEnergyReferenceDirectionFactory(8, [3, 2])

ref_dirs = fac.do()


ref_dirs = RandomSamplingAndMap(n_dim, n_points, seed=1).do()
Scatter(title="Random Sampling + Mapping").add(ref_dirs).show()

ref_dirs = ConstructionBasedReferenceDirectionFactory(n_dim, n_points, seed=1).do()
Scatter(title="Construction").add(ref_dirs).show()

ref_dirs = ReductionBasedReferenceDirectionFactory(n_dim, n_points, seed=1).do()
Scatter(title="Reduction").add(ref_dirs).show()

ref_dirs = RieszEnergyReferenceDirectionFactory(n_dim, n_points, seed=1).do()
Scatter(title="Riesz s-Energy").add(ref_dirs).show()


