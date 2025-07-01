import sys


BACKENDS = {
    "numpy": "numpy",
    "autograd": "autograd.numpy", 
    "jax": "jax.numpy"
}

# Current active backend
active_backend = "numpy"

def activate(name):
    global active_backend
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Choose from: {list(BACKENDS.keys())}")
    active_backend = name
    # No need to delete module - LazyBackend handles dynamic switching


def deactivate():
    activate("numpy")


