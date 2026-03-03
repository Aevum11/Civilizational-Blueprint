"""
Exception Theory - A Comprehensive Mathematical Framework

Exception Theory is a complete ontological framework built on three fundamental primitives:
- P (Point): The substrate of existence
- D (Descriptor): Constraints and properties
- T (Traverser): Agency and navigation

These combine to form: PDT = EIM = S
(Point-Descriptor-Traverser = Exception-Incoherence-Mediation = Something)

From: "For every exception there is an exception, except the exception."

Author: Derived from M.J.M.'s Exception Theory
Version: 3.0.0
"""

__version__ = "3.0.0"
__author__ = "M.J.M. (Exception Theory) / ET Development Team (Implementation)"
__license__ = "MIT"

# Core imports
from .core import (
    # Mathematics
    ETMathV2,
    
    # Primitives
    PrimitiveType,
    Point,
    Descriptor,
    Traverser,
    Exception as ETException,
    bind_pdt,
    create_point,
    create_descriptor,
    create_traverser,
    
    # Constants (all exposed from core.constants)
    BASE_VARIANCE,
    MANIFOLD_SYMMETRY,
    KOIDE_RATIO,
    VERSION,
)

# Class imports (all batches)
from .classes import (
    # Batch 1: Computational Exception Theory
    TraverserEntropy,
    TrinaryState,
    ChameleonObject,
    TraverserMonitor,
    RealityGrounding,
    TemporalCoherenceFilter,
    EvolutionarySolver,
    PNumber,
    
    # Batch 2: Advanced Manifold Architectures
    TeleologicalSorter,
    ProbabilisticManifold,
    HolographicValidator,
    ZeroKnowledgeProtocol,
    ContentAddressableStorage,
    ReactivePoint,
    GhostSwitch,
    UniversalAdapter,
    
    # Batch 3: Distributed Consciousness
    SwarmConsensus,
    PrecognitiveCache,
    ImmortalSupervisor,
    SemanticManifold,
    VarianceLimiter,
    ProofOfTraversal,
    EphemeralVault,
    ConsistentHashingRing,
    TimeTraveler,
    FractalReality,
)

# Engine import
from .engine import ETSovereign

# Utilities
from .utils import (
    ETBeaconField,
    ETContainerTraverser,
    get_logger,
    set_log_level,
    enable_debug,
    enable_info,
    enable_warning,
    disable_logging,
)

__all__ = [
    # Version
    '__version__',
    '__author__',
    '__license__',
    
    # Core - Mathematics
    'ETMathV2',
    
    # Core - Primitives
    'PrimitiveType',
    'Point',
    'Descriptor',
    'Traverser',
    'ETException',
    'bind_pdt',
    'create_point',
    'create_descriptor',
    'create_traverser',
    
    # Core - Key Constants
    'BASE_VARIANCE',
    'MANIFOLD_SYMMETRY',
    'KOIDE_RATIO',
    'VERSION',
    
    # Classes - Batch 1
    'TraverserEntropy',
    'TrinaryState',
    'ChameleonObject',
    'TraverserMonitor',
    'RealityGrounding',
    'TemporalCoherenceFilter',
    'EvolutionarySolver',
    'PNumber',
    
    # Classes - Batch 2
    'TeleologicalSorter',
    'ProbabilisticManifold',
    'HolographicValidator',
    'ZeroKnowledgeProtocol',
    'ContentAddressableStorage',
    'ReactivePoint',
    'GhostSwitch',
    'UniversalAdapter',
    
    # Classes - Batch 3
    'SwarmConsensus',
    'PrecognitiveCache',
    'ImmortalSupervisor',
    'SemanticManifold',
    'VarianceLimiter',
    'ProofOfTraversal',
    'EphemeralVault',
    'ConsistentHashingRing',
    'TimeTraveler',
    'FractalReality',
    
    # Engine
    'ETSovereign',
    
    # Utilities
    'ETBeaconField',
    'ETContainerTraverser',
    'get_logger',
    'set_log_level',
    'enable_debug',
    'enable_info',
    'enable_warning',
    'disable_logging',
]
