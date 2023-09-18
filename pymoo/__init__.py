from pymoo.version import __version__
import numpy as np
import threading

# Create the global random state Singleton
class PymooPRNG(object):
   _lock = threading.Lock()
   _instance = None

   @classmethod
   def instance(cls, seed=None):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = np.random.default_rng(seed)
                    
        return cls._instance
