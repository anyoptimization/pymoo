from pymoo.version import __version__
import numpy as np

# Create the global random state
# NOTE do not use from to import c.f. https://stackoverflow.com/a/15959638/5517459
# TODO use builtin to define the global prng? 
PYMOO_PRNG = np.random.default_rng()
