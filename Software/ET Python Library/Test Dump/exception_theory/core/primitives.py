"""
Exception Theory Primitives Module

Defines the three fundamental primitives of Exception Theory:
- P (Point): The substrate of existence, the "where" of Something
- D (Descriptor): Constraints and properties, the "what" of Something  
- T (Traverser): Agency and navigation, the "who" of Something

These primitives combine to form E (Exception) = I (Incoherence) = M (Mediation) = S (Something)

PDT = EIM = S (The Master Identity)

From: "For every exception there is an exception, except the exception."

Author: Derived from M.J.M.'s Exception Theory
"""

from typing import Any, Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum, auto


class PrimitiveType(Enum):
    """The three fundamental primitives of Exception Theory."""
    POINT = auto()        # P - Substrate, Location, Potential
    DESCRIPTOR = auto()   # D - Constraint, Property, Limitation
    TRAVERSER = auto()    # T - Agency, Navigation, Choice


@dataclass
class Point:
    """
    P (Point): The substrate of existence.
    
    In ET: P is pure potential, location without property.
    A Point can hold any configuration of Descriptors.
    
    Attributes:
        location: Identifier or coordinate in the manifold
        state: Current state/value at this Point
        descriptors: Set of Descriptors bound to this Point
    """
    location: Any
    state: Optional[Any] = None
    descriptors: Optional[List['Descriptor']] = None
    
    def bind(self, descriptor: 'Descriptor') -> 'Point':
        """
        Bind a Descriptor to this Point.
        P ∘ D creates a constrained configuration.
        
        Args:
            descriptor: Descriptor to bind
        
        Returns:
            Self for chaining
        """
        if self.descriptors is None:
            self.descriptors = []
        self.descriptors.append(descriptor)
        return self
    
    def substantiate(self, value: Any) -> 'Point':
        """
        Substantiate this Point with a concrete value.
        Moves from potential to actual.
        
        Args:
            value: Value to substantiate
        
        Returns:
            Self for chaining
        """
        self.state = value
        return self


@dataclass
class Descriptor:
    """
    D (Descriptor): Constraints and properties.
    
    In ET: D limits what can exist at a Point.
    Descriptors create structure and order from pure potential.
    
    Attributes:
        name: Identifier for this Descriptor
        constraint: The constraint function or value
        metadata: Additional information about the constraint
    """
    name: str
    constraint: Any
    metadata: Optional[Dict[str, Any]] = None
    
    def apply(self, point: Point) -> bool:
        """
        Apply this Descriptor's constraint to a Point.
        
        Args:
            point: Point to constrain
        
        Returns:
            True if Point satisfies this constraint, False otherwise
        """
        if callable(self.constraint):
            return self.constraint(point.state)
        else:
            return point.state == self.constraint
    
    def compose(self, other: 'Descriptor') -> 'Descriptor':
        """
        Compose two Descriptors: D₁ ∘ D₂
        
        Args:
            other: Descriptor to compose with
        
        Returns:
            New composed Descriptor
        """
        def composed_constraint(value):
            return self.constraint(value) and other.constraint(value)
        
        return Descriptor(
            name=f"{self.name}∘{other.name}",
            constraint=composed_constraint,
            metadata={'composition': (self, other)}
        )


@dataclass
class Traverser:
    """
    T (Traverser): Agency and navigation.
    
    In ET: T is the principle of choice and movement through the manifold.
    T is inherently indeterminate - it represents free will and unpredictability.
    
    Attributes:
        identity: Identifier for this Traverser
        current_point: Current location in the manifold
        history: Path history of Points traversed
    """
    identity: str
    current_point: Optional[Point] = None
    history: Optional[List[Point]] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
    
    def traverse(self, target_point: Point) -> 'Traverser':
        """
        Navigate from current Point to target Point.
        T navigates the manifold, making choices.
        
        Args:
            target_point: Destination Point
        
        Returns:
            Self for chaining
        """
        if self.current_point is not None:
            self.history.append(self.current_point)
        self.current_point = target_point
        return self
    
    def observe(self, point: Point) -> Any:
        """
        Observe a Point's state.
        T-observation collapses superposition.
        
        Args:
            point: Point to observe
        
        Returns:
            The observed state
        """
        return point.state
    
    def choose(self, options: List[Point], criterion: Any = None) -> Point:
        """
        Make a choice between multiple Points.
        This is where T's indeterminacy manifests.
        
        Args:
            options: List of available Points
            criterion: Optional selection criterion (else random)
        
        Returns:
            Chosen Point
        """
        if criterion is not None:
            # Deterministic choice based on criterion
            return max(options, key=criterion)
        else:
            # Indeterminate choice (T-singularity)
            import random
            return random.choice(options)


class Exception:
    """
    E (Exception): The unified state PDT.
    
    In ET: E = P ∘ D ∘ T = Something that violates a constraint.
    Everything that exists is an Exception to void.
    
    The Exception is the fundamental unit of reality.
    """
    
    def __init__(self, point: Point, descriptor: Descriptor, traverser: Optional[Traverser] = None):
        """
        Create an Exception from primitives.
        
        Args:
            point: The Point (substrate)
            descriptor: The Descriptor (constraint)
            traverser: The Traverser (agency) - optional
        """
        self.point = point
        self.descriptor = descriptor
        self.traverser = traverser
    
    def is_coherent(self) -> bool:
        """
        Check if this Exception is coherent.
        A coherent Exception satisfies all its constraints.
        
        Returns:
            True if coherent, False if incoherent
        """
        return self.descriptor.apply(self.point)
    
    def substantiate(self) -> Tuple[Point, Descriptor, Optional[Traverser]]:
        """
        Decompose the Exception back into primitives.
        
        Returns:
            Tuple of (P, D, T)
        """
        return (self.point, self.descriptor, self.traverser)


def bind_pdt(point: Point, descriptor: Descriptor, traverser: Optional[Traverser] = None) -> Exception:
    """
    P ∘ D ∘ T = E
    
    The Master Equation binding operator.
    Creates an Exception from the three primitives.
    
    Args:
        point: P - The substrate
        descriptor: D - The constraint
        traverser: T - The agency (optional)
    
    Returns:
        Exception instance
    """
    return Exception(point, descriptor, traverser)


# Convenience functions for creating primitives

def create_point(location: Any, initial_state: Any = None) -> Point:
    """Create a Point at the given location."""
    return Point(location=location, state=initial_state)


def create_descriptor(name: str, constraint: Any, **metadata) -> Descriptor:
    """Create a Descriptor with the given constraint."""
    return Descriptor(name=name, constraint=constraint, metadata=metadata or None)


def create_traverser(identity: str, starting_point: Optional[Point] = None) -> Traverser:
    """Create a Traverser with the given identity."""
    return Traverser(identity=identity, current_point=starting_point)


__all__ = [
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
