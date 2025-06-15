"""
Visualization module for pymoo.

All matplotlib functionality should be accessed through:
    from pymoo.visualization.matplotlib import plt, patches, etc.

This ensures proper handling when matplotlib is not available.
"""

# Import centralized matplotlib module
from . import matplotlib

__all__ = ["heatmap",
           "pcp", 
           "petal",
           "radar",
           "radviz",
           "scatter",
           "star_coordinate",
           "matplotlib"
           ]
