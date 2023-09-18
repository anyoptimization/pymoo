from pymoo.version import __version__
import numpy as np
import threading

# Create the global random state Singleton
class PymooPRNG(object):
    _lock = threading.Lock()
    _instance = None

    def __new__(cls, seed=None):
        if cls._instance is None: 
            with cls._lock:
                # Another thread could have created the instance
                # before we acquired the lock. So check that the
                # instance is still nonexistent.
                if not cls._instance:
                    cls._instance = np.random.default_rng(seed)
                    
        return cls._instance
