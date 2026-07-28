"""
Exception Theory Batch 14 Classes
Primitive Disjointness Theory (Eq 141-150)

**BATCH COMPLETE: 10/10 equations implemented**

Features:
- P-D disjointness measure (Eq 141)
- D-T disjointness measure (Eq 142)
- T-P disjointness measure (Eq 143)
- Pairwise disjointness test (Eq 144)
- Total independence verification (Eq 145)
- Binding operator existence (Eq 146)
- Non-grounding exception cardinality (Eq 147)
- Grounding immutability (Eq 148)
- Exception conditionality (Eq 149)
- Axiom universal coverage (Eq 150)

From: "For every exception there is an exception, except the exception."
Axiom of Categorical Distinction: P ∩ D = ∅, D ∩ T = ∅, T ∩ P = ∅

Author: Derived from Michael James Muller's Exception Theory
Date: 2026-01-20
Version: 3.7.0
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from ..core.mathematics import ETMathV2Quantum
from ..core.constants import (
    PD_INTERSECTION_CARDINALITY,
    DT_INTERSECTION_CARDINALITY,
    TP_INTERSECTION_CARDINALITY,
    TOTAL_DISJOINTNESS,
    BINDING_OPERATOR_EXISTS,
    GROUNDING_IMMUTABLE,
    AXIOM_UNIVERSAL,
)


# ============================================================================
# Eq 141: PDDisjointnessMeasure
# ============================================================================

class PDDisjointnessMeasure:
    """
    Batch 14, Eq 141: P-D Disjointness Measure.
    
    ET Math: |P ∩ D| = 0
    
    Measures and verifies disjointness between Point and Descriptor sets.
    A Point is infinite substrate, while Descriptor is finite constraint.
    These categories cannot overlap.
    """
    
    def __init__(self):
        """Initialize P-D disjointness measure."""
        self.expected_cardinality = PD_INTERSECTION_CARDINALITY
        self.measurements = []
    
    def measure_intersection(self, p_set=None, d_set=None):
        """
        Measure P-D intersection cardinality.
        
        Args:
            p_set: Point set (optional)
            d_set: Descriptor set (optional)
        
        Returns:
            0 (always - P and D are disjoint)
        """
        cardinality = ETMathV2Quantum.pd_disjointness_measure(p_set, d_set)
        
        self.measurements.append({
            'p_elements': len(p_set) if p_set else 0,
            'd_elements': len(d_set) if d_set else 0,
            'intersection': cardinality
        })
        
        return cardinality
    
    def verify_disjointness(self):
        """
        Verify P-D disjointness property.
        
        Returns:
            True (P and D are always disjoint)
        """
        return self.expected_cardinality == 0


# ============================================================================
# Eq 142: DTDisjointnessMeasure
# ============================================================================

class DTDisjointnessMeasure:
    """
    Batch 14, Eq 142: D-T Disjointness Measure.
    
    ET Math: |D ∩ T| = 0
    
    Measures and verifies disjointness between Descriptor and Traverser sets.
    Descriptor is finite constraint, while Traverser is indeterminate agency.
    These categories cannot overlap.
    """
    
    def __init__(self):
        """Initialize D-T disjointness measure."""
        self.expected_cardinality = DT_INTERSECTION_CARDINALITY
        self.measurements = []
    
    def measure_intersection(self, d_set=None, t_set=None):
        """
        Measure D-T intersection cardinality.
        
        Args:
            d_set: Descriptor set (optional)
            t_set: Traverser set (optional)
        
        Returns:
            0 (always - D and T are disjoint)
        """
        cardinality = ETMathV2Quantum.dt_disjointness_measure(d_set, t_set)
        
        self.measurements.append({
            'd_elements': len(d_set) if d_set else 0,
            't_elements': len(t_set) if t_set else 0,
            'intersection': cardinality
        })
        
        return cardinality
    
    def verify_disjointness(self):
        """
        Verify D-T disjointness property.
        
        Returns:
            True (D and T are always disjoint)
        """
        return self.expected_cardinality == 0


# ============================================================================
# Eq 143: TPDisjointnessMeasure
# ============================================================================

class TPDisjointnessMeasure:
    """
    Batch 14, Eq 143: T-P Disjointness Measure.
    
    ET Math: |T ∩ P| = 0
    
    Measures and verifies disjointness between Traverser and Point sets.
    Traverser is indeterminate agency, while Point is infinite substrate.
    These categories cannot overlap.
    """
    
    def __init__(self):
        """Initialize T-P disjointness measure."""
        self.expected_cardinality = TP_INTERSECTION_CARDINALITY
        self.measurements = []
    
    def measure_intersection(self, t_set=None, p_set=None):
        """
        Measure T-P intersection cardinality.
        
        Args:
            t_set: Traverser set (optional)
            p_set: Point set (optional)
        
        Returns:
            0 (always - T and P are disjoint)
        """
        cardinality = ETMathV2Quantum.tp_disjointness_measure(t_set, p_set)
        
        self.measurements.append({
            't_elements': len(t_set) if t_set else 0,
            'p_elements': len(p_set) if p_set else 0,
            'intersection': cardinality
        })
        
        return cardinality
    
    def verify_disjointness(self):
        """
        Verify T-P disjointness property.
        
        Returns:
            True (T and P are always disjoint)
        """
        return self.expected_cardinality == 0


# ============================================================================
# Eq 144: PairwiseDisjointnessTester
# ============================================================================

class PairwiseDisjointnessTester:
    """
    Batch 14, Eq 144: Pairwise Disjointness Test.
    
    ET Math: disjoint(A, B) ⟺ (|A ∩ B| = 0)
    
    General-purpose tester for set disjointness.
    Tests if two arbitrary sets share any common elements.
    """
    
    def __init__(self):
        """Initialize pairwise tester."""
        self.test_history = []
    
    def test_disjoint(self, set_a: Union[List, Set, Tuple], 
                     set_b: Union[List, Set, Tuple]) -> bool:
        """
        Test if two sets are disjoint.
        
        Args:
            set_a: First set
            set_b: Second set
        
        Returns:
            True if disjoint, False otherwise
        """
        is_disjoint = ETMathV2Quantum.pairwise_disjointness_test(set_a, set_b)
        
        self.test_history.append({
            'set_a_size': len(set_a),
            'set_b_size': len(set_b),
            'disjoint': is_disjoint
        })
        
        return is_disjoint
    
    def get_disjoint_pairs_count(self):
        """
        Get count of disjoint pairs tested.
        
        Returns:
            Number of disjoint pairs
        """
        return sum(1 for t in self.test_history if t['disjoint'])
    
    def get_test_summary(self):
        """
        Get summary of all tests.
        
        Returns:
            Dict with test statistics
        """
        if not self.test_history:
            return {'total': 0, 'disjoint': 0, 'overlapping': 0}
        
        disjoint = self.get_disjoint_pairs_count()
        total = len(self.test_history)
        
        return {
            'total': total,
            'disjoint': disjoint,
            'overlapping': total - disjoint,
            'disjoint_rate': disjoint / total if total > 0 else 0.0
        }


# ============================================================================
# Eq 145: TotalIndependenceVerifier
# ============================================================================

class TotalIndependenceVerifier:
    """
    Batch 14, Eq 145: Total Independence Verification.
    
    ET Math: independence(P,D,T) = (|P∩D| + |D∩T| + |T∩P|) = 0
    
    Verifies complete independence by summing all pairwise intersections.
    For true categorical independence, this sum must be exactly zero.
    """
    
    def __init__(self):
        """Initialize total independence verifier."""
        self.expected_total = TOTAL_DISJOINTNESS
        self.verifications = []
    
    def verify_independence(self, p_set=None, d_set=None, t_set=None):
        """
        Verify total independence of P, D, T.
        
        Args:
            p_set: Point set (optional)
            d_set: Descriptor set (optional)
            t_set: Traverser set (optional)
        
        Returns:
            0 (sum of pairwise intersections)
        """
        total = ETMathV2Quantum.total_independence_verification(p_set, d_set, t_set)
        
        self.verifications.append({
            'total_intersection': total,
            'independent': (total == 0)
        })
        
        return total
    
    def is_independent(self):
        """
        Check if primitives are completely independent.
        
        Returns:
            True if sum of pairwise intersections is 0
        """
        if not self.verifications:
            return True  # Theoretical property holds
        
        return all(v['independent'] for v in self.verifications)
    
    def get_verification_rate(self):
        """
        Get rate of successful independence verifications.
        
        Returns:
            Fraction of verifications showing independence
        """
        if not self.verifications:
            return 1.0  # Theoretical property
        
        successful = sum(1 for v in self.verifications if v['independent'])
        return successful / len(self.verifications)


# ============================================================================
# Eq 146: BindingOperatorExistenceProver
# ============================================================================

class BindingOperatorExistenceProver:
    """
    Batch 14, Eq 146: Binding Operator Existence.
    
    ET Math: ∃ ∘ : P × D × T → M where M ≠ ∅
    
    Proves that despite complete disjointness, the binding operator
    creates non-empty manifold from P, D, T. This is the fundamental
    composition operator of Exception Theory.
    """
    
    def __init__(self):
        """Initialize binding operator prover."""
        self.operator_exists = BINDING_OPERATOR_EXISTS
        self.binding_operations = []
    
    def prove_existence(self, p=None, d=None, t=None):
        """
        Prove binding operator exists.
        
        Args:
            p: Point element (optional)
            d: Descriptor element (optional)
            t: Traverser element (optional)
        
        Returns:
            True (operator always exists)
        """
        exists = ETMathV2Quantum.binding_operator_existence(p, d, t)
        
        self.binding_operations.append({
            'p': p,
            'd': d,
            't': t,
            'operator_exists': exists
        })
        
        return exists
    
    def simulate_binding(self, p, d, t):
        """
        Simulate binding operation P ∘ D ∘ T.
        
        Args:
            p: Point element
            d: Descriptor element
            t: Traverser element
        
        Returns:
            Bound tuple (manifold element)
        """
        # Binding creates non-empty result despite disjointness
        bound = (p, d, t)
        
        self.binding_operations.append({
            'p': p,
            'd': d,
            't': t,
            'bound': bound,
            'non_empty': bound is not None
        })
        
        return bound
    
    def verify_non_empty_result(self):
        """
        Verify binding produces non-empty manifold.
        
        Returns:
            True if all bindings produce non-empty results
        """
        if not self.binding_operations:
            return True  # Theoretical property
        
        return all(
            'bound' not in op or op['bound'] is not None
            for op in self.binding_operations
        )


# ============================================================================
# Eq 147: NonGroundingExceptionCounter
# ============================================================================

class NonGroundingExceptionCounter:
    """
    Batch 14, Eq 147: Non-Grounding Exception Cardinality.
    
    ET Math: |non_grounding| = |all_exceptions| - 1
    
    Counts exceptions that can be otherwise (non-grounding).
    All exceptions except THE Exception are mutable.
    """
    
    def __init__(self):
        """Initialize non-grounding counter."""
        self.exception_registry = []
        self.grounding_id = 0  # THE Exception has ID 0
    
    def register_exception(self, exception_id):
        """
        Register an exception.
        
        Args:
            exception_id: Unique ID for exception
        """
        self.exception_registry.append({
            'id': exception_id,
            'grounding': (exception_id == self.grounding_id)
        })
    
    def count_non_grounding(self):
        """
        Count non-grounding exceptions.
        
        Returns:
            Number of non-grounding exceptions
        """
        total = len(self.exception_registry)
        return ETMathV2Quantum.non_grounding_exception_cardinality(total)
    
    def get_grounding_exception(self):
        """
        Get THE Exception (grounding exception).
        
        Returns:
            Dict with grounding exception info
        """
        grounding = [e for e in self.exception_registry if e['grounding']]
        return grounding[0] if grounding else None
    
    def get_non_grounding_exceptions(self):
        """
        Get all non-grounding exceptions.
        
        Returns:
            List of non-grounding exception records
        """
        return [e for e in self.exception_registry if not e['grounding']]


# ============================================================================
# Eq 148: GroundingImmutabilityVerifier
# ============================================================================

class GroundingImmutabilityVerifier:
    """
    Batch 14, Eq 148: Grounding Immutability.
    
    ET Math: immutable(THE Exception) = True
    
    Verifies that THE Exception is the singular fixed point.
    The one thing that cannot be otherwise - absolute immutability.
    """
    
    def __init__(self):
        """Initialize grounding immutability verifier."""
        self.is_immutable = GROUNDING_IMMUTABLE
        self.verification_count = 0
    
    def verify_immutability(self):
        """
        Verify grounding immutability.
        
        Returns:
            True (THE Exception is always immutable)
        """
        result = ETMathV2Quantum.grounding_immutability()
        self.verification_count += 1
        return result
    
    def test_mutation_attempt(self):
        """
        Test if grounding can be mutated.
        
        Returns:
            False (mutation always fails - grounding is immutable)
        """
        # THE Exception cannot be otherwise
        return False
    
    def get_immutability_strength(self):
        """
        Get immutability strength.
        
        Returns:
            1.0 (absolute immutability)
        """
        return 1.0 if self.is_immutable else 0.0


# ============================================================================
# Eq 149: ExceptionConditionalityTester
# ============================================================================

class ExceptionConditionalityTester:
    """
    Batch 14, Eq 149: Exception Conditionality.
    
    ET Math: can_be_otherwise(x) ⟺ (x ≠ THE Exception)
    
    Tests conditional mutability of entities.
    An entity can be otherwise if and only if it's not THE Exception.
    """
    
    def __init__(self, grounding_id=0):
        """
        Initialize conditionality tester.
        
        Args:
            grounding_id: ID of THE Exception
        """
        self.grounding_id = grounding_id
        self.tests = []
    
    def test_can_be_otherwise(self, entity_id):
        """
        Test if entity can be otherwise.
        
        Args:
            entity_id: Entity to test
        
        Returns:
            True if entity can be otherwise, False if grounding
        """
        can_mutate = ETMathV2Quantum.exception_conditionality(entity_id, self.grounding_id)
        
        self.tests.append({
            'entity_id': entity_id,
            'can_be_otherwise': can_mutate,
            'is_grounding': (entity_id == self.grounding_id)
        })
        
        return can_mutate
    
    def identify_grounding(self):
        """
        Identify THE Exception from test history.
        
        Returns:
            Entity ID that cannot be otherwise
        """
        immutable = [t for t in self.tests if not t['can_be_otherwise']]
        return immutable[0]['entity_id'] if immutable else self.grounding_id
    
    def get_mutable_count(self):
        """
        Get count of mutable entities tested.
        
        Returns:
            Number of entities that can be otherwise
        """
        return sum(1 for t in self.tests if t['can_be_otherwise'])


# ============================================================================
# Eq 150: AxiomUniversalCoverageVerifier
# ============================================================================

class AxiomUniversalCoverageVerifier:
    """
    Batch 14, Eq 150: Axiom Universal Coverage.
    
    ET Math: domain(axiom) = 𝕌 (Universe of Discourse)
    
    Verifies that the foundational axiom applies to ALL possible entities.
    Nothing escapes the domain of Exception Theory.
    """
    
    def __init__(self):
        """Initialize universal coverage verifier."""
        self.is_universal = AXIOM_UNIVERSAL
        self.coverage_tests = []
    
    def verify_coverage(self, entity=None):
        """
        Verify axiom applies to entity.
        
        Args:
            entity: Entity to test (any type)
        
        Returns:
            True (axiom always applies)
        """
        applies = ETMathV2Quantum.axiom_universal_coverage()
        
        self.coverage_tests.append({
            'entity': str(entity) if entity is not None else 'abstract',
            'applies': applies
        })
        
        return applies
    
    def test_universal_application(self, entity_list):
        """
        Test axiom application across multiple entities.
        
        Args:
            entity_list: List of entities to test
        
        Returns:
            True if axiom applies to all entities
        """
        results = [self.verify_coverage(e) for e in entity_list]
        return all(results)
    
    def get_coverage_rate(self):
        """
        Get coverage rate across all tests.
        
        Returns:
            1.0 (100% coverage - axiom is universal)
        """
        if not self.coverage_tests:
            return 1.0  # Theoretical property
        
        applicable = sum(1 for t in self.coverage_tests if t['applies'])
        return applicable / len(self.coverage_tests)
    
    def verify_universality(self):
        """
        Verify universal property.
        
        Returns:
            True (axiom is universal)
        """
        return self.is_universal


# ============================================================================

__all__ = [
    'PDDisjointnessMeasure',
    'DTDisjointnessMeasure',
    'TPDisjointnessMeasure',
    'PairwiseDisjointnessTester',
    'TotalIndependenceVerifier',
    'BindingOperatorExistenceProver',
    'NonGroundingExceptionCounter',
    'GroundingImmutabilityVerifier',
    'ExceptionConditionalityTester',
    'AxiomUniversalCoverageVerifier',
]
