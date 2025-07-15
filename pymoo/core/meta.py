from copy import deepcopy

import wrapt


class Meta(wrapt.ObjectProxy):
    """
    An object that automatically delegates all method calls to the wrapped object
    unless the method is explicitly overridden in the Meta subclass.
    """

    def __init__(self, wrapped_object, copy=True):
        if copy:
            wrapped_object = deepcopy(wrapped_object)
        
        # Initialize wrapt.ObjectProxy with the object
        super().__init__(wrapped_object)
        
        # Keep these attributes for backward compatibility
        self.__object__ = wrapped_object
        self.__super__ = wrapped_object
        self.__original__ = wrapped_object if not copy else None
        
    
    def __setattr__(self, name, value):
        # Store certain attributes on the proxy object itself, not the wrapped object
        if name in ['_self_', '__wrapped__', '__object__', '__super__', '__original__', 'cache'] or name.startswith('_self_'):
            super().__setattr__(name, value)
        else:
            # For all other attributes, store them on this object, not the wrapped one
            super().__setattr__(name, value)

    def __deepcopy__(self, memo):
        """Support deepcopy for wrapt.ObjectProxy objects."""
        from copy import deepcopy
        
        # For MetaProblem and its subclasses, use the simple approach
        if hasattr(self, '__wrapped__'):
            copied_wrapped = deepcopy(self.__wrapped__, memo)
            
            # Get the actual proxy class (not the wrapped object's class)
            proxy_class = type(self)
            
            # Create the new instance - try with specific arguments first, fallback to generic
            try:
                # Try class-specific constructors
                if 'AutomaticDifferentiation' in proxy_class.__name__ and hasattr(self, 'eps'):
                    copied_meta = proxy_class(copied_wrapped, eps=getattr(self, 'eps', 1e-8))
                elif 'ComplexNumberGradient' in proxy_class.__name__ and hasattr(self, 'eps'):
                    copied_meta = proxy_class(copied_wrapped, eps=getattr(self, 'eps', 1e-8))
                elif 'ConstraintsAsPenalty' in proxy_class.__name__ and hasattr(self, 'penalty'):
                    copied_meta = proxy_class(copied_wrapped, penalty=getattr(self, 'penalty', 0.1))
                elif 'ConstraintsAsObjective' in proxy_class.__name__ and hasattr(self, 'config'):
                    copied_meta = proxy_class(copied_wrapped, 
                                            config=getattr(self, 'config', None),
                                            append=getattr(self, 'append', True))
                elif 'ConstraintsFromBounds' in proxy_class.__name__ and hasattr(self, 'remove_bonds'):
                    copied_meta = proxy_class(copied_wrapped, remove_bonds=getattr(self, 'remove_bonds', False))
                elif 'StaticProblem' in proxy_class.__name__ and hasattr(self, 'kwargs'):
                    copied_meta = proxy_class(copied_wrapped, **getattr(self, 'kwargs', {}))
                else:
                    # Generic case - just pass the wrapped object
                    copied_meta = proxy_class(copied_wrapped)
            except TypeError:
                # If specific constructor fails, try generic
                copied_meta = proxy_class(copied_wrapped)
            
            memo[id(self)] = copied_meta
            return copied_meta
        
        # Fallback for other cases
        return deepcopy(self.__wrapped__, memo)

