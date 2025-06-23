"""
Patches for autograd to work with numpy 2.x
"""
import warnings

def patch_autograd_for_numpy2():
    """Patch autograd functions to handle numpy 2.x keyword arguments"""
    try:
        import autograd.numpy as anp
        from autograd.numpy.numpy_vjps import defvjp
        from autograd.core import primitive
        
        # Store original functions
        _original_sum = anp.sum
        
        # Create wrapper that filters out unsupported kwargs
        @primitive
        def patched_sum(x, axis=None, dtype=None, keepdims=False, initial=None, where=None, out=None):
            # Filter out unsupported kwargs for autograd
            kwargs = {'axis': axis, 'keepdims': keepdims}
            if dtype is not None:
                kwargs['dtype'] = dtype
            return _original_sum(x, **kwargs)
        
        # Replace the function
        anp.sum = patched_sum
        
        # Copy the VJP maker from the original
        if hasattr(_original_sum, 'vjpmaker'):
            patched_sum.vjpmaker = _original_sum.vjpmaker
            
    except ImportError:
        warnings.warn("autograd not installed, skipping numpy 2.x patches")