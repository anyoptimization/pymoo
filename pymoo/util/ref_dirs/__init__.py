"""Reference direction factories and utilities for multi-objective optimization."""

from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory
from pymoo.util.ref_dirs.energy_layer import (
    LayerwiseRieszEnergyReferenceDirectionFactory,
)
from pymoo.util.ref_dirs.reduction import ReductionBasedReferenceDirectionFactory
from pymoo.util.ref_dirs.incremental import IncrementalReferenceDirectionFactory
from pymoo.util.reference_direction import MultiLayerReferenceDirectionFactory

__all__ = [
    "RieszEnergyReferenceDirectionFactory",
    "LayerwiseRieszEnergyReferenceDirectionFactory",
    "ReductionBasedReferenceDirectionFactory",
    "IncrementalReferenceDirectionFactory",
    "MultiLayerReferenceDirectionFactory",
    "get_reference_directions",
]  # noqa: F401


def get_reference_directions(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
    """Factory function to instantiate reference direction generators by name.

    Args:
        name: Name of the reference direction factory ("uniform", "das-dennis", "energy",
            "multi-layer", "layer-energy", "reduction", or "incremental").
        *args: Positional arguments passed to the factory constructor.
        **kwargs: Keyword arguments passed to the factory constructor.

    Returns:
        An instantiated reference direction factory object.

    Raises:
        Exception: If the reference direction factory name is not found.
    """
    from pymoo.util.reference_direction import UniformReferenceDirectionFactory

    REF = {
        "uniform": UniformReferenceDirectionFactory,
        "das-dennis": UniformReferenceDirectionFactory,
        "energy": RieszEnergyReferenceDirectionFactory,
        "multi-layer": MultiLayerReferenceDirectionFactory,
        "layer-energy": LayerwiseRieszEnergyReferenceDirectionFactory,
        "reduction": ReductionBasedReferenceDirectionFactory,
        "incremental": IncrementalReferenceDirectionFactory,
    }

    if name not in REF:
        raise Exception("Reference directions factory not found.")

    return REF[name](*args, **kwargs)()
