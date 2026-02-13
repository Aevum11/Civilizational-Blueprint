"""
Exception Theory Core Module

This module contains the fundamental mathematics, primitives, and constants
for Exception Theory.

From: "For every exception there is an exception, except the exception."

Author: Derived from M.J.M.'s Exception Theory
"""

from .constants import *
from .mathematics import ETMathV2
from .primitives import (
    PrimitiveType,
    Point,
    Descriptor,
    Traverser,
    Exception,
    bind_pdt,
    create_point,
    create_descriptor,
    create_traverser,
)

__all__ = [
    # Mathematics
    'ETMathV2',
    
    # Primitives
    'PrimitiveType',
    'Point',
    'Descriptor',
    'Traverser',
    'Exception',
    'bind_pdt',
    'create_point',
    'create_descriptor',
    'create_traverser',
]

# Expose all constants without listing them individually
from .constants import __all__ as _constants_all
__all__.extend(_constants_all)
