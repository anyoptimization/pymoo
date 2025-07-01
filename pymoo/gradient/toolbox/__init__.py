import sys
import importlib

class LazyBackend:
    """Lazy backend that always uses the current active backend"""
    
    def __getattr__(self, name):
        from pymoo.gradient import active_backend, BACKENDS
        backend_module = importlib.import_module(BACKENDS[active_backend])
        return getattr(backend_module, name)
    
    def __dir__(self):
        """Support for tab completion and introspection"""
        from pymoo.gradient import active_backend, BACKENDS
        backend_module = importlib.import_module(BACKENDS[active_backend])
        return dir(backend_module)

# Replace this module with the lazy backend
sys.modules[__name__] = LazyBackend()
