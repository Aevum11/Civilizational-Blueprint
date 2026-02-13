"""
Exception Theory Batch 20: DESCRIPTOR NATURE & CARDINALITY (Eq 201-210)

Systematic extraction of Descriptor (D) primitive foundations covering:
- Absolute finitude and cardinality
- "How" ontology and constraint structure  
- Differentiation properties
- Binding effects and bounded values
- Framework priority to spacetime

All derived from Exception Theory primitives: P (Point), D (Descriptor), T (Traverser)

Author: Derived from Michael James Muller's Exception Theory
Version: 3.10.0
"""

from typing import Any, Dict, List, Optional, Set, Union
from ..core.constants import (
    DESCRIPTOR_IS_FINITE,
    DESCRIPTOR_IS_HOW,
    DESCRIPTOR_DIFFERENTIATES,
    DESCRIPTOR_BOUND_VALUES,
    FINITE_DESCRIPTION_WAYS,
    DESCRIPTOR_BOUND_TO_POINT,
    UNBOUND_DESCRIPTOR_INFINITE,
    BINDING_CREATES_FINITUDE,
    SPACETIME_IS_DESCRIPTOR,
    FRAMEWORK_PRIOR_SPACETIME,
)
from ..core.mathematics import ETMathV2


class DescriptorFinitudeAnalyzer:
    """Batch 20, Eq 201: |D| = n (Absolute Finite Cardinality)
    
    ET Math: cardinality(D) = n where n ∈ ℕ
    
    Analyzes the absolute finitude of descriptor sets.
    Despite infinite Points, descriptors are finite.
    """
    
    def __init__(self, descriptor_set: List[Any]):
        """
        Initialize analyzer with descriptor set.
        
        Args:
            descriptor_set: Collection of descriptors
        """
        self.descriptors = descriptor_set
        self.cardinality = ETMathV2.descriptor_absolute_finite(len(descriptor_set))
    
    def verify_finitude(self) -> Dict[str, Any]:
        """Verify that descriptor set has finite cardinality."""
        return {
            "cardinality": self.cardinality,
            "is_finite": self.cardinality < float('inf'),
            "descriptor_count": len(self.descriptors),
            "property": "absolute_finite"
        }
    
    def count_descriptors(self) -> int:
        """Get finite count of descriptors."""
        return self.cardinality


class DescriptorHowOntologyMapper:
    """Batch 20, Eq 202: D = "How" (Constraints and Structure)
    
    ET Math: D = constraint_specification
    
    Maps descriptors to their constraint/structure ontology.
    Descriptor is the "How" of existence.
    """
    
    def __init__(self, constraint_spec: Any):
        """
        Initialize ontology mapper.
        
        Args:
            constraint_spec: Specification of constraints
        """
        self.constraint = constraint_spec
        self.ontology = ETMathV2.descriptor_how_ontology(constraint_spec)
    
    def get_how_specification(self) -> Dict[str, Any]:
        """Get the "How" ontological specification."""
        return self.ontology
    
    def classify_constraint(self) -> str:
        """Classify the type of constraint."""
        if isinstance(self.constraint, (int, float)):
            return "numerical_constraint"
        elif isinstance(self.constraint, str):
            return "symbolic_constraint"
        elif isinstance(self.constraint, (list, tuple)):
            return "structural_constraint"
        else:
            return "complex_constraint"


class ConfigurationDifferentiator:
    """Batch 20, Eq 203: D(P₁) ≠ D(P₂) → P₁ ≠ P₂
    
    ET Math: Different descriptors imply different Points
    
    Analyzes how descriptors differentiate configurations.
    """
    
    def __init__(self, p1_descriptors: List[Any], p2_descriptors: List[Any]):
        """
        Initialize differentiator with two descriptor sets.
        
        Args:
            p1_descriptors: Descriptors for first Point
            p2_descriptors: Descriptors for second Point
        """
        self.p1_desc = p1_descriptors
        self.p2_desc = p2_descriptors
        self.are_different = ETMathV2.descriptor_differentiation(
            p1_descriptors, p2_descriptors
        )
    
    def configurations_differ(self) -> bool:
        """Check if configurations are different."""
        return self.are_different
    
    def find_differentiating_descriptors(self) -> List[Any]:
        """Find which descriptors differ between configurations."""
        p1_set = set(str(d) for d in self.p1_desc)
        p2_set = set(str(d) for d in self.p2_desc)
        
        differences = p1_set.symmetric_difference(p2_set)
        return list(differences)


class BoundedValueGenerator:
    """Batch 20, Eq 204: D_bound = P∘D → |D_bound| < ∞
    
    ET Math: Binding creates bounded values
    
    Generates bounded values through P-D binding.
    """
    
    def __init__(self, point: Optional[Any], descriptor: Optional[Any]):
        """
        Initialize with Point and Descriptor.
        
        Args:
            point: Point substrate
            descriptor: Descriptor constraint
        """
        self.point = point
        self.descriptor = descriptor
        self.bounded_value = ETMathV2.descriptor_bounded_by_binding(
            point, descriptor
        )
    
    def get_bounded_value(self) -> Union[int, float]:
        """Get the bounded value from binding."""
        return self.bounded_value
    
    def is_finite(self) -> bool:
        """Check if value is finite."""
        return self.bounded_value < float('inf')
    
    def binding_status(self) -> Dict[str, Any]:
        """Get binding status information."""
        return {
            "point_present": self.point is not None,
            "descriptor_present": self.descriptor is not None,
            "bounded_value": self.bounded_value,
            "is_finite": self.is_finite()
        }


class FiniteDescriptionCalculator:
    """Batch 20, Eq 205: |{D : P∘D}| = n
    
    ET Math: Finite ways to describe any Point
    
    Calculates finite description possibilities.
    """
    
    def __init__(self, point: Any, descriptor_space: List[Any]):
        """
        Initialize calculator with Point and descriptor space.
        
        Args:
            point: The Point to describe
            descriptor_space: Available descriptors
        """
        self.point = point
        self.descriptor_space = descriptor_space
        self.description_count = ETMathV2.finite_description_ways(
            point, descriptor_space
        )
    
    def count_descriptions(self) -> int:
        """Count finite ways to describe the Point."""
        return self.description_count
    
    def is_describable(self) -> bool:
        """Check if Point has at least one description."""
        return self.description_count > 0


class DescriptorBindingEnforcer:
    """Batch 20, Eq 206: ∃P : P∘D (D cannot exist without P)
    
    ET Math: Every descriptor must bind to a Point
    
    Enforces binding necessity for descriptors.
    """
    
    def __init__(self, descriptor: Any):
        """
        Initialize enforcer with descriptor.
        
        Args:
            descriptor: The descriptor to check
        """
        self.descriptor = descriptor
        self.binding_required = ETMathV2.descriptor_binding_necessity(descriptor)
    
    def requires_binding(self) -> bool:
        """Check if descriptor requires Point binding."""
        return self.binding_required
    
    def validate_binding(self, point: Optional[Any]) -> Dict[str, Any]:
        """Validate binding to a Point."""
        is_valid = point is not None and self.descriptor is not None
        return {
            "descriptor_exists": self.descriptor is not None,
            "point_exists": point is not None,
            "binding_valid": is_valid,
            "status": "bound" if is_valid else "unbound"
        }


class UnboundInfinityDetector:
    """Batch 20, Eq 207: ¬∃P : P∘D → |D| = ∞
    
    ET Math: Unbound descriptors collapse to infinity
    
    Detects infinity in unbound descriptors.
    """
    
    def __init__(self, is_bound: bool):
        """
        Initialize detector.
        
        Args:
            is_bound: Whether descriptor is bound to Point
        """
        self.is_bound = is_bound
        self.cardinality = ETMathV2.unbound_descriptor_infinity(is_bound)
    
    def is_infinite(self) -> bool:
        """Check if descriptor has infinite cardinality."""
        return self.cardinality == float('inf')
    
    def get_cardinality(self) -> float:
        """Get descriptor cardinality."""
        return self.cardinality


class BindingFinitudeTransformer:
    """Batch 20, Eq 208: P∘D → |D| < ∞
    
    ET Math: Binding transforms infinite to finite
    
    Transforms descriptor through binding to finitude.
    """
    
    def __init__(self, point: Optional[Any], descriptor: Any):
        """
        Initialize transformer.
        
        Args:
            point: Point substrate (None if unbound)
            descriptor: Descriptor value
        """
        self.point = point
        self.descriptor = descriptor
        self.finite_value = ETMathV2.binding_collapse_to_finite(
            point, descriptor
        )
    
    def transform_to_finite(self) -> Union[int, float]:
        """Get finite value after binding transformation."""
        return self.finite_value
    
    def transformation_report(self) -> Dict[str, Any]:
        """Report on the transformation process."""
        return {
            "input_descriptor": str(self.descriptor),
            "point_bound": self.point is not None,
            "output_value": self.finite_value,
            "is_finite": self.finite_value < float('inf'),
            "transformation": "infinity_to_finite" if self.point else "remains_infinite"
        }


class SpacetimeDescriptorClassifier:
    """Batch 20, Eq 209: time, space, causality, laws ⊂ D
    
    ET Math: Spacetime properties are Descriptors
    
    Classifies spacetime properties as descriptors.
    """
    
    def __init__(self, property_name: str):
        """
        Initialize classifier.
        
        Args:
            property_name: Name of property to classify
        """
        self.property = property_name
        self.is_spacetime = ETMathV2.spacetime_as_descriptor(property_name)
    
    def classify(self) -> Dict[str, Any]:
        """Classify whether property is spacetime descriptor."""
        return {
            "property": self.property,
            "is_spacetime_descriptor": self.is_spacetime,
            "category": "spacetime" if self.is_spacetime else "other",
            "descriptor_status": "valid_descriptor"
        }
    
    def get_spacetime_type(self) -> Optional[str]:
        """Get specific spacetime type if applicable."""
        if not self.is_spacetime:
            return None
        
        prop_lower = self.property.lower()
        if 'time' in prop_lower:
            return "temporal"
        elif 'space' in prop_lower or 'position' in prop_lower:
            return "spatial"
        elif 'causal' in prop_lower or 'sequence' in prop_lower:
            return "causal"
        elif 'law' in prop_lower:
            return "lawful"
        return "general_spacetime"


class FrameworkPriorityAnalyzer:
    """Batch 20, Eq 210: P∘D∘T precedes spacetime emergence
    
    ET Math: ET framework exists before spacetime
    
    Analyzes ontological priority of ET framework.
    """
    
    def __init__(self):
        """Initialize priority analyzer."""
        self.priority_structure = ETMathV2.framework_prior_spacetime()
    
    def get_priority_order(self) -> List[str]:
        """Get ontological priority ordering."""
        return self.priority_structure["order"]
    
    def verify_priority(self) -> Dict[str, Any]:
        """Verify framework priority over spacetime."""
        return {
            "framework": "P∘D∘T",
            "emergent": "spacetime",
            "priority": self.priority_structure["priority"],
            "ontological_order": self.priority_structure["order"],
            "framework_first": True
        }
    
    def explain_priority(self) -> str:
        """Explain why framework precedes spacetime."""
        return (
            "Exception Theory (P∘D∘T) is ontologically prior to spacetime "
            "because spacetime itself is a Descriptor - a constraint on "
            "Point configurations. The framework must exist before its "
            "descriptive properties can emerge."
        )


__all__ = [
    'DescriptorFinitudeAnalyzer',
    'DescriptorHowOntologyMapper',
    'ConfigurationDifferentiator',
    'BoundedValueGenerator',
    'FiniteDescriptionCalculator',
    'DescriptorBindingEnforcer',
    'UnboundInfinityDetector',
    'BindingFinitudeTransformer',
    'SpacetimeDescriptorClassifier',
    'FrameworkPriorityAnalyzer',
]
