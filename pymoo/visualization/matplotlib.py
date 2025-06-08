"""
Centralized matplotlib imports for pymoo visualization.

This module provides a single point of entry for all matplotlib functionality,
with graceful handling when matplotlib is not available.
"""

# Try to import matplotlib and related modules
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    from matplotlib import animation
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.colors import ListedColormap
    
    # Set backend and configuration
    matplotlib.use('Agg', force=False)  # Use non-interactive backend by default
    
    _MATPLOTLIB_AVAILABLE = True
    
    # Export all commonly used matplotlib objects
    __all__ = [
        'matplotlib', 'plt', 'patches', 'colors', 'cm', 'animation', 
        'LineCollection', 'PatchCollection', 'ListedColormap', 'is_available'
    ]
    
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    
    class _MatplotlibNotAvailable:
        """Helper class that raises informative errors when matplotlib is not available."""
        
        def __getattr__(self, name):
            raise ImportError(
                "Visualization features require matplotlib.\n"
                "Install with: pip install pymoo[visualization]"
            )
        
        def __call__(self, *args, **kwargs):
            raise ImportError(
                "Visualization features require matplotlib.\n"
                "Install with: pip install pymoo[visualization]"
            )
    
    # Create placeholder objects that give helpful errors
    matplotlib = _MatplotlibNotAvailable()
    plt = _MatplotlibNotAvailable()
    patches = _MatplotlibNotAvailable()
    colors = _MatplotlibNotAvailable()
    cm = _MatplotlibNotAvailable()
    animation = _MatplotlibNotAvailable()
    LineCollection = _MatplotlibNotAvailable()
    PatchCollection = _MatplotlibNotAvailable()
    ListedColormap = _MatplotlibNotAvailable()


def is_available():
    """Check if matplotlib is available for visualization."""
    return _MATPLOTLIB_AVAILABLE


